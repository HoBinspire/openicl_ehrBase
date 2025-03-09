"""Topk Retriever"""

from openicl import DatasetReader
from openicl.icl_dataset_reader import DatasetEncoder
from openicl.icl_retriever import BaseRetriever
from openicl.utils.collators import DataCollatorWithPaddingAndCuda
from openicl.utils.logging import get_logger
import torch
from torch.utils.data import DataLoader
from typing import Optional
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import tqdm
import faiss
import copy
import numpy as np
from accelerate import Accelerator

logger = get_logger(__name__)


class TopkRetriever(BaseRetriever):
    """Topk In-context Learning Retriever Class
        Class of Topk Retriever.
        
    Attributes:
        dataset_reader (:obj:`DatasetReader`): An instance of the :obj:`DatasetReader` class.
        ice_separator (:obj:`str`, optional): A string that separates each in-context example.
        ice_eos_token (:obj:`str`, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (:obj:`str`, optional): A string that is added to the end of the prompt.
        ice_num (:obj:`int`, optional): The number of data in the in-context examples.
        index_split (:obj:`str`, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. Defaults to ``train``.
        test_split (:obj:`str`, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. Defaults to ``test``.
        index_ds (:obj:`Dataset`): The index dataset. Used to select data for in-context examples.
        test_ds (:obj:`Dataset`): The test dataset. Used to generate prompts for each data.
        accelerator (:obj:`Accelerator`, optional): An instance of the :obj:`Accelerator` class, used for multiprocessing.
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`. 
        model (:obj:`SentenceTransformer`): An instance of :obj:`SentenceTransformer` class, used to calculate embeddings.
        tokenizer (:obj:`AutoTokenizer`): Tokenizer for :obj:`model`.
        index (:obj:`IndexIDMap`): Index generated with FAISS.
    """
    model = None

    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 sentence_transformers_model_name: Optional[str] = '/data/lhb/huggingface/model/sentence_embedding/all-mpnet-base-v2',
                 ice_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 tokenizer_name: Optional[str] = '/data/lhb/huggingface/model/tokenizer/gpt2-xl',
                 batch_size: Optional[int] = 1,
                 accelerator: Optional[Accelerator] = None
                 ) -> None:
        # 父类初始化：多进程分片 和 self.index_dx， self.text_ds 的提取
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, ice_num, index_split,
                         test_split, accelerator)
        
        # 初始化 设备、分词器、数据加载器
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name

        # 配置分词器 参数（填充符， 截断方向）
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        # 测试数据 语料加载器
        gen_datalist = self.dataset_reader.generate_input_field_corpus(self.test_ds)  # 拼接测试集中 所有测试样例的 输入特征 返回 测试语料库 list[str]
        self.encode_dataset = DatasetEncoder(gen_datalist, tokenizer=self.tokenizer)  # self.encode_dataset.encode_dataset 是一个 encode 结果，类型为 list[dict] ?????????????????????????????????这里有点奇怪，感觉需要修改
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)  # 数据整理器：将多个单独数据样本 填充到相同长度，组成一个 batch, 将填充后的张量 自动转移到指定设备
        self.dataloader = DataLoader(self.encode_dataset, batch_size=self.batch_size, collate_fn=co)  # 输出加载器：每次加载一个 batch, 每次加载 batch 调用 co

        # 加载 句子嵌入模型
        self.model = SentenceTransformer(sentence_transformers_model_name)
        self.model = self.model.to(self.device)
        self.model.eval()  # 使用时，设置为 评估模式

        # 创建 faiss 索引
        self.index = self.create_index()

    def create_index(self):
        self.select_datalist = self.dataset_reader.generate_input_field_corpus(self.index_ds)  # 训练语料库 list[str], 每个 str 包含 特征列 和 label 列
        
        # 训练数据 语料加载器
        encode_datalist = DatasetEncoder(self.select_datalist, tokenizer=self.tokenizer)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)
        dataloader = DataLoader(encode_datalist, batch_size=self.batch_size, collate_fn=co)

        # 创建索引
        index = faiss.IndexIDMap(faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension()))  # 根据向量的维度；IndexFlatIP 表示使用 内积作为距离度量(单位向量 内积 等于 余弦相似度); IndexIDMap 为每个向量分配 唯一 ID 且 返回 `向量+id`
        
        # 进行 嵌入，提取 id
        res_list = self.forward(dataloader, process_bar=True, information="Creating index for index set...")  # return: [{"embed":..., "metadata":...}]
        id_list = np.array([res['metadata']['id'] for res in res_list])  # 提取 meta 信息中的 id
        self.embed_list = np.stack([res['embed'] for res in res_list])  # 用 stack 方法，将 embedding 向量 在一个新的维度 堆叠成 多一个维度的数组
        
        # 将向量矩阵 形状为 (n_samples, embed_dim) & id_list 形状为 (n_samples,) 添加到索引
        index.add_with_ids(self.embed_list, id_list)
        return index

    def knn_search(self, ice_num):
        """
        1. 调用 forward 生成测试集的嵌入向量; self.dataloader 是 测试语料 加载器；
        2. 对每个 测试样本 的嵌入向量，返回 ice_num 个 最近邻 训练样本的 id;
        3. 返回：每个测试样本对应一个索引列表，包含相似样本的ID
        """
        res_list = self.forward(self.dataloader, process_bar=True, information="Embedding test set...")  # 测试集的 嵌入+id 列表
        
        rtr_idx_list = [[] for _ in range(len(res_list))]  # 初始化 result
        logger.info("Retrieving data for test set...")

        for entry in tqdm.tqdm(res_list, disable=not self.is_main_process):
            idx = entry['metadata']['id']
            embed = np.expand_dims(entry['embed'], axis=0) # 由于 search函数参数 embed 需要是二维数组，这里增加一个维度
            near_ids = self.index.search(embed, ice_num)[1][0].tolist()  # search 函数返回 ([[0.48833227, 0.40074888, 0.34541786]], [[3046, 3045, 6865]])，第一个数组是 精度计算结果，第二个是向量 对应的 id 
            rtr_idx_list[idx] = near_ids  # meta 中的 id 一定是从 0开始的自然数，这里没问题
        return rtr_idx_list

    def forward(self, dataloader, process_bar=False, information=''):
        """
        1. 将分词后的 input_ids 解码为原始文本
        2. 生成嵌入：使用 Sentence Transformer 对文本编码。
        3. 保存结果：存储嵌入向量 及其 元数据（如数据ID）
        """
        res_list = []
        _dataloader = copy.deepcopy(dataloader)
        if process_bar:  # 是否显示进度条
            logger.info(information)
            _dataloader = tqdm.tqdm(_dataloader, disable=not self.is_main_process)
        for _, entry in enumerate(_dataloader):  # entry 是一个 字典类型数据 {'input_ids':[[]],  'attention_mask': [[]],    'metadata': {'id': 0, 'len': 6, 'text': 'I get up early today.'}}
            with torch.no_grad():
                metadata = entry.pop("metadata")
                raw_text = self.tokenizer.batch_decode(entry['input_ids'], skip_special_tokens=True, verbose=False)  # 将 input_ids 解码为 原始文本
                res = self.model.encode(raw_text, show_progress_bar=False)  # 生成 原始文本的 嵌入
            res_list.extend([{"embed": r, "metadata": m} for r, m in zip(res, metadata)])  # 将 文本嵌入 + metadata 打包  加入 list
        return res_list

    def retrieve(self):
        """
        返回 list[list], 表示 测试集中 每条数据 最相似的 ice_num 条数据 在 训练集中的 id
        """
        return self.knn_search(self.ice_num)
