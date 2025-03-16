import time
from utils import process_pdf
import argparse

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Process PDF document')
    parser.add_argument('--input_pdf', required=True, help='Path to input PDF file')
    parser.add_argument('--api_key', required=True, help='API key for service authentication')
    parser.add_argument('--pages', type=int, default=100, help='Number of pages to process (default: 100)')

    # 解析参数
    args = parser.parse_args()

    # 执行处理并计时
    start_time = time.time()
    process_pdf(
        input_pdf=args.input_pdf,
        api_key=args.api_key,
        pages=args.pages
    )
    print(f"Execution time: {time.time() - start_time:.2f} seconds")
    # test_cmd="""python main.py --input_pdf pdf_path --api_key xxx --pages 100"""


