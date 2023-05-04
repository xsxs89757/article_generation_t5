以下是一个中文版的`README.md`文件：

```
# T5 文章生成

本项目演示了如何使用文章标题和内容的数据集对 T5 模型进行微调，以实现文章生成。项目还包括一个根据提供的标题生成文章的 API。

## 开始

### 先决条件

- Python 3.7+
- PyTorch 1.10.0+
- Hugging Face Transformers 4.11.3+


### 安装

1. 克隆仓库：

   ```
   git clone https://github.com/xsxs89757/article_generation_t5.git
   cd article_generation_t5
   ```

2. 创建虚拟环境：

   - Windows：
   
     ```
     python -m venv myenv
     myenv\Scripts\activate
     ```

   - macOS 和 Linux：
   
     ```
     python3 -m venv myenv
     source myenv/bin/activate
     ```

3. 安装所需库：

   ```
   pip install -r requirements.txt
   ```

4. 使用 `environment.yml` 安装所需库（步骤3的替代方案）：

   ```
   conda env create -f environment.yml
   conda activate article_generation_t5
   ```

## 使用

1. 训练模型 T5：

   ```
   python src/train.py
   ```
2. 训练模型 T5 and DeepSpeed：

   ```
   deepspeed src/train_deepspeed.py --deepspeed --deepspeed_config ds_config.json
   ```

3. 使用训练好的模型生成文章：

   ```
   python src/generate.py
   ```

4. 启动 API：

   ```
   uvicorn src/api:app --reload
   ```

## 许可

本项目根据 MIT 许可证进行许可 - 请参阅 [LICENSE](LICENSE) 文件了解详情。
```