from datasets import load_dataset
from openicl import DatasetReader
from openicl import PromptTemplate
from openicl import TopkRetriever
from openicl.icl_retriever.icl_cone_retriever import ConERetriever
from openicl import PPLInferencer
from openicl import AccEvaluator

# 加载数据
dataset = load_dataset('/data/lhb/huggingface/dataset/sst2_gpt3mix')
data = DatasetReader(dataset, input_columns=['text'], output_column='label')

# 构造模板
tp_dict = {
    0: "</E>Positive Movie Review: </text>",
    1: "</E>Negative Movie Review: </text>" 
}
template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

# 构建检索器
# retriever = TopkRetriever(data, ice_num=8)
retriever = ConERetriever(data, candidate_num = 30, ice_num=8)

# 构建推理器
inferencer = PPLInferencer(model_name='/data/lhb/huggingface/model/tokenizer/gpt2-xl', output_json_filepath = "./icl_inference_output", output_json_filename = "predictions")

# 推理
predictions = inferencer.inference(retriever, ice_template=template)

# 评估
score = AccEvaluator().score(predictions=predictions, references=data.references)
print(score)
