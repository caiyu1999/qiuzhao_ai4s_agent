from langchain.utils.strings import collapse_whitespace

raw_text = "物 拟 人 算 法 设 计 ． R e s e a r c h B a c k g r o u n d"
cleaned = collapse_whitespace(raw_text)  # 基础清理