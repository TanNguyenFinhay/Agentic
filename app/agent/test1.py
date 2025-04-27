from reasoner import Reasoner  # Giả sử bạn lưu Reasoner class ở file reasoner.py

def main():
    reasoner = Reasoner()

    prompt = input("Nhập câu hỏi để lên kế hoạch: ")

    # Step 1: Lập kế hoạch
    plan = reasoner.plan(prompt)
    print("\n=== Plan ===")
    print(plan)

    # Step 2: Refine kế hoạch (giả lập dữ liệu context)
    file_context = "Thông tin từ tài liệu nội bộ."
    web_context = "Kết quả tìm kiếm từ web."
    refined_plan = reasoner.refine(plan, file_context, web_context)
    print("\n=== Refined Plan ===")
    print(refined_plan)

    # Step 3: Reflect báo cáo (giả lập báo cáo ban đầu)
    raw_report = "Bản báo cáo thô dựa trên refined plan."
    final_report = reasoner.reflect(raw_report)
    print("\n=== Final Report ===")
    print(final_report)

if __name__ == "__main__":
    main()