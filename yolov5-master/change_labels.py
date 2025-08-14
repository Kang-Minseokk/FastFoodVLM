import os
import re

def change_hamburger_labels(labels_path='data/food_image/labels/val'):
    """
    hamburger{숫자}.txt 파일들의 클래스 번호를 0에서 5로 변경
    """
    
    # 경로 존재 확인
    if not os.path.exists(labels_path):
        print(f"경로를 찾을 수 없습니다: {labels_path}")
        return
    
    changed_files = 0
    total_lines_changed = 0
    total_hamburger_files = 0
    
    # 디렉토리 내의 모든 파일에 대해 반복
    for filename in os.listdir(labels_path):
        # hamburger로 시작하고 .txt로 끝나는 파일만 처리
        if filename.startswith('hamburger') and filename.endswith('.txt'):
            total_hamburger_files += 1
            filepath = os.path.join(labels_path, filename)
            
            # 파일 읽기
            with open(filepath, 'r') as file:
                lines = file.readlines()
            
            lines_changed_in_file = 0
            
            # 각 라인의 첫 번째 요소(클래스 번호)를 0에서 5로 변경
            for i in range(len(lines)):
                if lines[i].strip():  # 빈 줄이 아닌 경우
                    parts = lines[i].strip().split()
                    if parts :
                        parts[0] = '5'
                        lines[i] = ' '.join(parts) + '\n'
                        lines_changed_in_file += 1
            
            # 변경사항이 있는 경우에만 파일에 다시 쓰기
            if lines_changed_in_file > 0:
                with open(filepath, 'w') as file:
                    file.writelines(lines)
                
                changed_files += 1
                total_lines_changed += lines_changed_in_file
                print(f"{filename}: {lines_changed_in_file}줄 변경")
            else:
                print(f"{filename}: 변경할 내용 없음 (클래스 0이 없음)")
    
    print(f"\n변경 완료!")
    print(f"전체 hamburger 파일 수: {total_hamburger_files}")
    print(f"실제 변경된 파일 수: {changed_files}")
    print(f"총 변경된 라인 수: {total_lines_changed}")

if __name__ == "__main__":
    change_hamburger_labels()