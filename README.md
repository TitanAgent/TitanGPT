TitanGPT 

这个项目是一个本地运行的问答应用程序，它结合了大型语言模型（LLM）和向量数据库，让你可以在本地使用自己的知识库（如书籍、文档等）进行问答，完全无需网络连接。
项目亮点
完全离线：所有处理都在本地完成，数据安全有保障。
可自定义知识库：通过简单的脚本，你可以用自己的文本文件（如.txt格式）来构建专属知识库。
模块化设计：项目分为两个主要脚本：一个用于构建数据库，另一个用于进行问答，结构清晰，易于扩展。
快速开始
1. 准备工作
在开始之前，请确保你已经安装了 Python 环境，并且具备以下文件或文件夹：
chatglm3-6b/ 文件夹：包含 ChatGLM3-6B 模型文件。你可以从官方渠道下载。
你想用来构建知识库的 .txt 格式的文本文件。
2. 安装依赖
在终端中，进入项目目录，然后运行以下命令来安装所有必需的 Python 库：
pip install torch transformers langchain-community langchain-huggingface faiss-cpu


3. 构建你的知识库
运行 shentan.py 脚本来处理你的文本文件并创建向量数据库。这个数据库会存储在 titan_db 文件夹中。
# 处理一个文件
python shentan.py your_book.txt

# 处理多个文件
python shentan.py book1.txt book2.txt another_doc.txt


注意：首次运行时，shentan.py 会创建 titan_db 文件夹。如果你添加新文件，它会智能地将新内容追加到现有数据库中，不会重复处理。
4. 启动本地问答程序
当数据库创建完成后，你可以运行 titanai.py 脚本来启动问答程序。
python titanai.py


程序启动后，你会看到一个命令行提示符。输入你的问题，程序会根据你的知识库内容提供回答。
项目结构
/TitanGPT
├── titanai.py          # 问答主程序
├── shentan.py          # 向量数据库构建工具
├── .gitignore          # Git 忽略文件配置
├── README.md           # 本文件
├── chatglm3-6b/        # ChatGLM3-6B 模型文件夹 (需手动下载)
└── titan_db/           # 向量数据库文件夹 (由 shentan.py 生成)


贡献
如果你有任何改进建议或发现 Bug，欢迎提出。这个项目是一个学习和实践的良好平台，期待你的贡献！

许可证
本代码遵循 MIT 许可证。
