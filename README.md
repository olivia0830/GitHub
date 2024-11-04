
# GitHub
from transformers import pipeline

# 加载预训练的问答模型
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def ai_customer_service(query, context):
    """
    简单的AI智能客服函数，使用预训练的问答模型回答用户问题。
    
    参数:
    query -- 用户的问题
    context -- 提供给模型的上下文，用于找到问题的答案
    
    返回:
    模型预测的答案
    """
    result = qa_model({
        'question': query,
        'context': context
    })
    return result['answer']

# 使用示例
if __name__ == "__main__":
    # 提供上下文，例如产品描述或常见问题解答
    context = """
    GitHub is a development platform inspired by the way you work. From open source to business,
    you can host and review code, manage projects, and build software alongside millions of other developers.
    """
    
    # 用户提问
    user_query = "What is GitHub?"

    # 获取答案
    answer = ai_customer_service(user_query, context)
    
    print(f"Answer: {answer}")
