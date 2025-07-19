from pydantic import BaseModel,Field


class ResponseFormatter_template_rewrite(BaseModel):
    """
    用于生成代码修改的结构化输出模型。
    - answer: 对用户问题的回答
    - rewrite_code: 重写的程序代码
    """
    
    suggestion: str = Field(default="",description="The suggestion to the user's question")
    rewrite_code: str = Field(default="",
        description="""
# Task
Rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs
as the original program, but with improved internal implementation.

```{language}
# Your rewritten program here
```""")

class ResponseFormatter_template_rewrite_with_web(BaseModel):
    """
    用于生成代码修改的结构化输出模型。
    - answer: 对用户问题的回答
    - rewrite_code: 重写的程序代码
    - diff_code: 用于指示代码修改的diff格式
    """
    
    suggestion: str = Field(default="",description="The suggestion to the user's question")
    rewrite_code: str = Field(default="",
        description="""
# Task
Rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.
If necessary, you can search the website to find the most relevant information to the user's question and use it to improve the program.
IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs
as the original program, but with improved internal implementation.

```{language}
# Your rewritten program here
```""")
 
class ResponseFormatter_template_diff(BaseModel):
    """
    用于生成代码修改的结构化输出模型。
    - answer: 对用户问题的回答
    - diff_code: 用于指示代码修改的diff格式
    """
    
    suggestion: str = Field(default="",description="The suggestion to the user's question")
    diff_code: str = Field(description="""
You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Example of valid diff format:
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.
Be thoughtful about your changes and explain your reasoning thoroughly.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.""")

class ResponseFormatter_template_diff_with_web(BaseModel):
    """
    用于生成代码修改的结构化输出模型。
    - answer: 对用户问题的回答
    - diff_code: 用于指示代码修改的diff格式
    """
    
    suggestion: str = Field(default="",description="The suggestion to the user's question")
    diff_code: str = Field(description="""
You MUST use the exact SEARCH/REPLACE diff format shown below to indicate changes:

<<<<<<< SEARCH
# Original code to find and replace (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE

Example of valid diff format:
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

You can suggest multiple changes. Each SEARCH section must exactly match code in the current program.
Be thoughtful about your changes and explain your reasoning thoroughly.

If necessary, you can search the website to find the most relevant information to the user's question and use it to improve the program.
IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.""")
    
class ResponseFormatter_template_evaluator(BaseModel):
    """
    用于评估代码质量的结构化输出模型。
    - readability: 代码的可读性评分(0.0-1.0)
    - maintainability: 代码的可维护性评分(0.0-1.0)
    - efficiency: 代码的效率评分(0.0-1.0)
    # - reasoning: 对评分的简要解释
    """
    readability: float = Field(default=0.0,description="The readability score of the code(0.0-1.0), the higher the score, the more readable the code",ge=0.0,le=1.0)
    maintainability: float = Field(default=0.0,description="The maintainability score of the code(0.0-1.0), the higher the score, the more maintainable the code",ge=0.0,le=1.0)
    efficiency: float = Field(default=0.0,description="The efficiency score of the code(0.0-1.0), the higher the score, the more efficient the code",ge=0.0,le=1.0)
    # reasoning: str = Field(default="",description="The short reasoning for the scores")