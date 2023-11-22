# DPSDA
[DPSDA Github](https://github.com/microsoft/DPSDA) 참고

## 모달리티 및 API 추가 시 변경해야 할 것들
* *!!* 가 달려 있는 항목은 API별 한 번씩 작업해야 함.
* *!!* 가 달려 있지 않는 항목은 모달리티별 한 번씩 작업해야 함.

* `main.py`
  1. `parse_args()`의 **api(29-34 lines)** 에서 choice에 사용할 API 이름 추가 *!!*
  2. `parse_args()`의 **modality(24-28 lines)** 에서 choice에 모달리티 추가
  3. `log_samples()`의 **(221~ lines)** 에서 모달리티에 따른 샘플 저장 방법 정의
  4. **Computing histogram 다음 부분** 에서 `visualize()`의 사용 여부 결정
  5. `visualize()`에서 다중 샘플 시각화 방법 정의
   
* `dpsda/dataset.py`
  1. **EXTENSIONS(7~ lines)** 정의
  2. 모달리티에 따른 **Dataset 클래스** 정의
  
* `dpsda/data_loader.py`
  1. `load_data()`의 **(13~ lines)** 에서 모달리티에 따른 dataset 정의
  2. `load_data()`의 **for문 아래로 첫 번째 if** 에서 모달리티에 따른 tensor 처리 과정 정의
   
* `dpsda/feature_extractor.py`
  1. `extract_features()`의 **(27~ lines)** 에서 모달리티에 따른 feature extract 방법 정의

* `apis/` 
  1. 사용할 API 클래스 정의 *!!*
     1. 기존의 정의된 파일을 기반으로 `__init__()`에서 필요한 변수 정의 - `__init__()`이 인자로 받는 모든 변수는 `command_line_parser()`에서 정의된 argument여야 함.
     2. `random_sampling()` 동작 방법 정의 - 내부 for문 위주로 변경하면 됨.
     3. `variation()` 동작 방법 정의 - variation degree가 정의되지 않는 API의 경우 DPSDA 논문에 따라 variation degree를 정수로 설정하고 그 크기만큼 반복 호출하도록 정의하였음. 추후 논의 필요.

* `apis/__init__.py`
  1. `get_api_class_from_name`에서 사용할 API 클래스 개체를 반환해주는 조건문 추가 *!!*
