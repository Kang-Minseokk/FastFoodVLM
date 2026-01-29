# check.py
import json, struct, sys
from pathlib import Path

def safetensors_index(path: str):
    """
    safetensors 파일의 헤더(JSON)만 읽어서
    각 텐서의 shape/dtype을 반환합니다. (데이터 디코딩 X)
    """
    path = Path(path)
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = f.read(header_len)
    meta = json.loads(header)
    out = {}
    for k, v in meta.items():
        if k == "__metadata__":
            continue
        out[k] = (tuple(v["shape"]), v["dtype"])
    return out

def diff(a, b):
    ka, kb = set(a.keys()), set(b.keys())
    only_a = sorted(ka - kb)
    only_b = sorted(kb - ka)
    common = sorted(ka & kb)

    changed = []
    for k in common:
        if a[k] != b[k]:
            changed.append((k, a[k], b[k]))
    return only_a, only_b, changed

def main():
    if len(sys.argv) != 3:
        print("Usage: python check.py ORIG/model.safetensors FT/model.safetensors")
        sys.exit(1)

    orig_path, ft_path = sys.argv[1], sys.argv[2]
    orig = safetensors_index(orig_path)
    ft   = safetensors_index(ft_path)

    only_orig, only_ft, changed = diff(orig, ft)

    print(f"orig tensors: {len(orig)} | ft tensors: {len(ft)}")
    print(f"only in orig: {len(only_orig)} | only in ft: {len(only_ft)} | changed: {len(changed)}\n")

    # 변경된 것 중 384/768이 껴있는 것만 먼저 출력
    print("=== CHANGED (contains 384 or 768) ===")
    cnt = 0
    for k, a, b in changed:
        a_shape, a_dtype = a
        b_shape, b_dtype = b
        if (384 in a_shape) or (768 in a_shape) or (384 in b_shape) or (768 in b_shape):
            print(f"{k}\n  orig: shape={a_shape}, dtype={a_dtype}\n  ft  : shape={b_shape}, dtype={b_dtype}\n")
            cnt += 1
            if cnt >= 200:
                print("(truncated at 200)")
                break

    # 전체 changed가 궁금하면 아래 주석 해제
    # print("=== ALL CHANGED (first 200) ===")
    # for i,(k,a,b) in enumerate(changed[:200]):
    #     print(k, a, "->", b)

if __name__ == "__main__":
    main()
