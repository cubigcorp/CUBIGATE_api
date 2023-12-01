# Data

* 데이터셋별로 디렉터리 생성
  * train/test/valid 구분은 서브 디렉터리로 처리
  * 클래스마다 별도의 디렉터리를 구성하거나 하나로 합쳐도 무관
* 데이터의 이름은 *Class이름_식별이름* 의 형태여야 함
  * Class이름을 추가하는 것은 `../preprocess.py` 참고
* 하나의 테이블에 모여 있는 데이터를 개별 파일로 바꾸는 작업이 필요하다면 `table_into_files.py` 참고
  
  ```
  table_into_files.py --table_file [TABLE_FILE] --result_dir [RESULT_DIR] --cols [COLUMNS] --label_col [LABEL_COLUMN] --train --test 
  ```

  * TABLE_FILE: 대상 테이블 파일의 경로
  * RESULT_DIR: 개별 파일을 저장할 디렉터리 경로, 해당 경로 하위에 train 또는 test 디렉터리가 생성되어 그 안에 개별 파일들이 저장됨
  * COLUMNS: 파일로 저장할 컬럼 목록, 띄어쓰기 없이 개별 컬럼은 콤마로 구분함
  * LABEL_COLUMN: 선택적, 컬럼 목록 중에서 라벨로 사용할 컬럼, 지정하지 않을 경우 UNCOND로 처리함
  * train, test: 대상 테이블 파일이 학습용인지 테스트용인지에 따라 둘 중 하나만 사용