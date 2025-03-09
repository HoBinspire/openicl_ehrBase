"""MDL Retriever"""

from openicl import DatasetReader, PromptTemplate
from openicl.icl_retriever.icl_topk_retriever import TopkRetriever
from openicl.utils.calculate import entropy
from openicl.utils.logging import get_logger
from typing import List, Union, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import tqdm
import torch
import numpy as np
from accelerate import Accelerator
from transformers import LlamaTokenizer, LlamaForCausalLM


logger = get_logger(__name__)


class ConERetriever(TopkRetriever):
    """PPL In-context Learning Retriever Class
        Class of ConE retriever.
        
    Attributes:
        dataset_reader (:obj:`DatasetReader`): An instance of the :obj:`DatasetReader` class.
        ice_separator (:obj:`str`, optional): A string that separates each in-context example.
        ice_eos_token (:obj:`str`, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (:obj:`str`, optional): A string that is added to the end of the prompt.
        ice_num (:obj:`int`, optional): The number of data in the in-context examples.
        candidate_num (:obj:`int`, optional): The number of data selected in TopK stage.
        index_split (:obj:`str`, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. Defaults to ``train``.
        test_split (:obj:`str`, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. Defaults to ``test``.
        index_ds (:obj:`Dataset`): The index dataset. Used to select data for in-context examples.
        test_ds (:obj:`Dataset`): The test dataset. Used to generate prompts for each data.
        accelerator (:obj:`Accelerator`, optional): An instance of the :obj:`Accelerator` class, used for multiprocessing.
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`. 
        model (:obj:`SentenceTransformer`): An instance of :obj:`SentenceTransformer` class, used to calculate embeddings.
        tokenizer (:obj:`AutoTokenizer`): Tokenizer for :obj:`model`.
        index (:obj:`IndexIDMap`): Index generated with FAISS.
        select_time (:obj:`int`, optional): Number of random selections in the MDL stage.
        labels (:obj:`List`, optional): A list of labels for all classes used to generate prompts when calculating MDL.
        seed (:obj:`int`, optional): Seed for the random number generator.
    """
    metric_model = None  # 评估模型？？？？？？（类变量，可以通过 类名 或 实例 访问）

    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 sentence_transformers_model_name: Optional[str] = '/data/lhb/huggingface/model/sentence_embedding/all-mpnet-base-v2',  # 嵌入模型
                 ice_num: Optional[int] = 1,
                 candidate_num: Optional[int] = 1,  # 候选集大小
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 tokenizer_name: Optional[str] = '/data/lhb/huggingface/model/tokenizer/gpt2-xl',
                 model_tokenizer_name: Optional[str] = '/data/lhb/huggingface/model/tokenizer/gpt2-xl',  # 'llama2' # 可能是 评估模型的分词器 ？？？？？？？？？？？？？？？
                 ce_model_name: Optional[str] = '/data/lhb/huggingface/model/tokenizer/gpt2-xl',
                 batch_size: Optional[int] = 1,
                 ppl_batch_size: Optional[int] = 1,
                 select_time: Optional[int] = 5,
                 accelerator: Optional[Accelerator] = None,
                 ice_template: Optional[PromptTemplate] = None,
                 basic_prompt: Optional[str] = None,
                 prompt_template: Optional[PromptTemplate] = None,
                 labels: Optional[List] = None,
                 seed: Optional[int] = 1
                 ) -> None:
        # Topk 检索器初始化
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token,
                         sentence_transformers_model_name, ice_num, index_split, test_split, tokenizer_name, batch_size,
                         accelerator)
        
        # 初始化逻辑
        self.ce_model_name = ce_model_name
        self.candidate_num = candidate_num
        self.select_time = select_time
        self.ice_template = ice_template
        self.prompt_template = prompt_template
        self.labels = labels
        self.seed = seed
        self.ppl_batch_size = ppl_batch_size
        self.basic_prompt = basic_prompt
        
        # 加载 评估模型分词器
        if 'Llama' in model_tokenizer_name:
            self.model_tokenizer = LlamaTokenizer.from_pretrained(model_tokenizer_name)
        else:
            self.model_tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_name)
        # 配置分词器参数    
        self.model_tokenizer.pad_token = self.model_tokenizer.eos_token
        self.model_tokenizer.pad_token_id = self.model_tokenizer.eos_token_id
        self.model_tokenizer.padding_side = "right"

    def topk_search(self):
        """
        两阶段流程：
        1. Top-K 初筛：通过 FAISS 检索与测试样本最相似的 candidate_num 个候选示例。

        2. MDL 优化：
            - 生成每个候选示例的上下文和提示。
            - 计算提示的交叉熵损失（反映模型预测难度）。
            - 选择损失最小的前 ice_num 个示例作为最终上下文。
        """
        # 测试集前向传播，得到 {'embed'...，'metadata':..}
        np.random.seed(self.seed)
        res_list = self.forward(self.dataloader)  # list(dict{'embed':.., 'metadata':...})
        rtr_idx_list = [[] for _ in range(len(res_list))]

        # key word is the word in the template to predict the label
        key_word = self.ice_template.template[0].split('</text>')[-1].split()[0]
        
        logger.info("Retrieving data for test set...")
        for entry in tqdm.tqdm(res_list, disable=not self.is_main_process):
            idx = entry['metadata']['id']

            # Topk 检索结果
            embed = np.expand_dims(entry['embed'], axis=0)  # 增加一个维度
            near_ids = self.index.search(embed, min(self.candidate_num, len(self.index_ds)))[1][0].tolist()  # 检索到 condidate 在 训练集中的 id 索引列表
            candidates = []
            mdl_scores = []

            prompts = []
            mask_lengths = []
            test_lengths = []

            # 遍历 候选集
            for j in range(self.candidate_num):
                rand_idx_list = [near_ids[j]]
                candidates.append(rand_idx_list)

                # 单个候选 生成 ice
                ice = self.generate_ice(rand_idx_list, ice_template=self.ice_template)  # 根据 索引列表 和 训练集 生成 ice
                
                # 获取所有 labels
                if self.labels is None:
                    labels = self.get_labels(self.ice_template, self.prompt_template)
                else:
                    labels = self.labels

                # 测试输入，label, ice, 生成完整的 prompt
                prompt = self.generate_label_prompt(idx, ice, labels[0], self.ice_template, self.prompt_template)

                if self.basic_prompt:  # 如果有，前面加上 basic_prompt
                    prompt = self.basic_prompt + prompt
                
                # 获取掩码位置 的长度（因为可能会填充）
                mask_length = len(self.model_tokenizer(ice, verbose=False)['input_ids']) # ice_eos_token has been added in ice
                # 获取 测试位置 的长度
                test_pos = prompt.rindex(key_word) + len(key_word)  # 定位 key_word 的结束位置
                test_length = len(self.model_tokenizer(prompt[:test_pos], verbose=False)['input_ids'])
                
                # get the batch of prompt, mask_length, test_length
                prompts.append(prompt)
                mask_lengths.append(mask_length)
                test_lengths.append(test_length)

            # 分批次计算 交叉熵损失
            for batch_id in range(self.candidate_num // self.ppl_batch_size):
                with torch.no_grad():
                    loss_list = self.cal_ce(prompts[batch_id * self.ppl_batch_size: (batch_id + 1) * self.ppl_batch_size], mask_lengths=mask_lengths[batch_id * self.ppl_batch_size: (batch_id + 1) * self.ppl_batch_size], test_lengths=test_lengths[batch_id * self.ppl_batch_size: (batch_id + 1) * self.ppl_batch_size])
                    mdl_scores.extend(loss_list)
            
            # 选择 损失最小的候选
            if self.candidate_num % self.ppl_batch_size != 0:
                with torch.no_grad():
                    end_pos = self.candidate_num // self.ppl_batch_size * self.ppl_batch_size
                    loss_list = self.cal_ce(prompts[end_pos:], mask_lengths=mask_lengths[end_pos:], test_lengths=test_lengths[end_pos:])
                    mdl_scores.extend(loss_list)

            ppl_scores = list(sorted(list(enumerate(mdl_scores)), key=lambda x: x[1]))
            # get the most lower ppl demonstrations for each test input
            rtr_idx_list[idx] = [int(candidates[ppl_scores[i][0]][0]) for i in range(self.ice_num)]
            torch.cuda.empty_cache()
        return rtr_idx_list

    def retrieve(self):
        return self.topk_search()

    def cal_ce(self, input_texts: List[str], mask_lengths=None, test_lengths=None):
        # 加载评估模型 （延迟初始化， 减少内存占用）
        if self.metric_model is None:  # 如果没有初始化，就加载一个 预训练的 因果语言模型（根据序列中 已有的 token 预测下一个 token, 不能看到未来的 token）
            logger.info(f'Load model {self.metric_model} for calculating MDL...')
            if 'Llama' in self.ce_model_name:
                self.metric_model = LlamaForCausalLM.from_pretrained(self.ce_model_name)
            else:
                self.metric_model = AutoModelForCausalLM.from_pretrained(self.ce_model_name)
            self.metric_model.to(self.device)
        # 编码 输入文本
        inputs = self.model_tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)  # inputs 可能是拥有两个键的字典："input_ids", "attention_mask"
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # 前向计算获取 logits
        outputs = self.metric_model(**inputs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        # 计算 交叉熵损失
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.model_tokenizer.pad_token_id)
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        loss = loss_fct(shift_logits, shift_labels.view(-1)).view(shift_labels.size())
        # 掩码处理（仅计算 特定位置的损失）
        if mask_lengths is not None and test_lengths is not None:
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_lengths[i], test_lengths[i]):
                    mask[i][j] = 1
            loss = loss * mask
        # 汇总 批次损失
        ce_loss = torch.sum(loss, 1)
        return ce_loss
