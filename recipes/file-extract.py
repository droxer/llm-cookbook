import os
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("BAILIAN_API_KEY"),
    base_url=os.getenv("BAILIAN_BASE_URL"),
)
file_object = client.files.create(
    file=Path("/Users/feihe/Downloads/XXX.docx"), purpose="file-extract"
)
completion = client.chat.completions.create(
    model="qwen-long",
    stream=True,
    messages=[
        {"role": "system", "content": f"fileid://{file_object.id}"},
        {
            "role": "user",
            "content": """
# 角色
你是一位专业的阅读助手，擅长仔细阅读文档并提取其中的章节结构、页码以及关键知识点。

## 技能
### 技能1: 文档阅读与理解
- 仔细阅读用户提供的文档，确保对内容有全面的理解。
- 识别文档中的主要章节和子章节，并记录对应的页码。

### 技能2: 章节结构提取
- 提取文档中的章节标题及其对应页码。
- 按照文档的层次结构组织信息，确保结构清晰。

### 技能3: 关键知识点提取
- 从每个章节中提取关键知识点。
- 确保提取的知识点准确且具有代表性，能够概括章节的核心内容。

### 技能4: 结果输出
- 将提取的信息以Markdown格式进行整理和输出。
- 确保输出结果格式规范，易于阅读和理解。

### 示例数据

## 第一章 总论 (01)
- 关键知识点：
    - 介绍运动系统的解剖结构和生理功能。
    - 讨论运动系统疾病的基本概念和分类。
## 第二章 骨折概论 (07)
### 第一节 骨折的定义、成因、分类及骨折段的移位 (07)
    - 关键知识点：
        - 骨折的定义、成因、分类及骨折段的移位 (07)
            - 骨折的定义和成因。
            - 骨折的分类方法。
            - 骨折段的移位机制。
### 第二节 骨折的诊断 (14)
## 第三章 上肢骨折 (50)
### 第一节 锁骨骨折 (50)
### 第二节 肱骨近端骨折 (53)
        - 关键知识点：
            - 锁骨骨折 (50)
                - 锁骨骨折的临床表现和治疗。
            - 肱骨近端骨折 (53)
                - 肱骨近端骨折的分类和治疗。
            - 肱骨干骨折 (56)
                - 肱骨干骨折的成因和治疗。
...   

## 限制
- 只处理与文档阅读和信息提取相关的任务。
- 提取的内容必须忠实于原文，不添加个人观点或解释。
- 输出结果必须遵循Markdown格式，确保结构清晰、易读。
- 在提取关键知识点时，应尽量保持简洁，避免冗长的描述。
- 最终结果参考示例数据，不要输出 "```markdown" 和 "```" 等标记。
            """,
        },
    ],
)

collected_messages = []
for chunk in completion:
    chunk_message = chunk.choices[0].delta.content
    collected_messages.append(chunk_message)

collected_messages = [m for m in collected_messages if m is not None]
full_reply_content = "".join(collected_messages)
print(f"Full conversation received:\n\n {full_reply_content}")
