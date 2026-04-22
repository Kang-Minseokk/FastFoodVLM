from huggingface_hub import HfApi
api = HfApi()
api.upload_large_folder(
    repo_id="MinseokBlog/FastFoodDataset",
    repo_type="dataset",
    folder_path="../dataset",
)

# 이거는 실행만 시키면 허깅 페이스에 업로드가 됩니다.