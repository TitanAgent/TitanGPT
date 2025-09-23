import os
import sys
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 确保模型和数据库路径正确
# 使用 os.path.join 拼接相对路径，这样可以跨平台工作
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "chatglm3-6b")
DB_PATH = os.path.join(BASE_DIR, "titan_db")

# 加载分词器和模型
def load_model_and_tokenizer():
    """
    加载 ChatGLM3-6B 模型和分词器。
    """
    print("🚀 正在加载 ChatGLM3-6B 模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).half().cuda()
        model = model.eval()
        print("✅ 模型加载成功！")
        return tokenizer, model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 请确保你已经将 chatglm3-6b 模型文件夹放置在项目根目录。")
        sys.exit(1)

# 加载向量数据库
def load_vector_db():
    """
    加载 FAISS 向量数据库。
    """
    print("📦 正在加载向量数据库...")
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh")
    try:
        db = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
        print("✅ 数据库加载成功！")
        return db
    except Exception as e:
        print(f"❌ 数据库加载失败: {e}")
        print("💡 请确保你已经运行过 shentan.py 来创建数据库。")
        sys.exit(1)

# 进行问答
def chat_with_model(db, tokenizer, model):
    """
    启动一个问答循环。
    """
    print("\n--- 本地问答程序已启动 ---")
    print("输入你的问题，或输入 'exit' 退出。")

    while True:
        query = input("\n👤 你: ")
        if query.lower() == 'exit':
            break

        if not query:
            continue

        print("🤖 正在思考中...")

        # 在数据库中检索相关文档
        docs = db.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        # 构建发送给模型的prompt
        prompt = f"基于以下知识库回答问题。\n知识库: {context}\n问题: {query}\n回答:"

        # 生成回答
        response, history = model.chat(tokenizer, prompt, history=[])
        print(f"🤖 TitanGPT: {response}")

# 主程序入口
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型路径不存在: {MODEL_PATH}")
        print("💡 请确保你已经将 chatglm3-6b 模型文件夹放置在项目根目录。")
        sys.exit(1)

    if not os.path.exists(DB_PATH):
        print(f"❌ 数据库路径不存在: {DB_PATH}")
        print("💡 请确保你已经运行过 shentan.py 来创建 'titan_db' 文件夹。")
        sys.exit(1)
        
    tokenizer, model = load_model_and_tokenizer()
    db = load_vector_db()

    chat_with_model(db, tokenizer, model)

    print("\n程序已退出。")
