# DPSDA
[DPSDA Github](https://github.com/microsoft/DPSDA) 참고

## 2024.01.04 Updates
### Parameters
* `dp`: DP 적용 여부, default=True
* `random_seed`: 난수 조절, default=2024
* `sample_weight`: sample-based variation에 대한 가중치, default=1.0
* `demonstration`: variation 시 demostration으로 보여줄 샘플의 개수, default=0
* `direct_variate`: 1226 그거 적용 여부, default=False
* `adaptive_variation_degree`: 샘플마다 variation degree를 다르게 설정할지 여부, default=False
* `diversity_lower_bound`: 패자부할전을 진행하지 않기 위한 다양성 하한선, default = 0.5
* `loser_lower_bound`: 패자로 분류되기 위한 투표수 하한선, default = N_syn / k

### Setting examples
* Vanila DPSDA: 위의 파라미터 모두 default로
* sample-based: `direct_variate true`
* demonstration-based: `direct_variate true`, `sample_weight 0`, `demonstration [DEMO]`
* mixed: `direct_variate true`, `sample_weight [W]`, `demonstration [DEMO]`
  * sample_weight < 1일 때 demonstration이 0이면 assert error
* non-DP: `dp false`
  * epsilon, delta, threshold는 따로 설정하였어도 0으로 변경
  * `direct_variate`와 `adaptive_variation_degree`도 True로 변경

---

## 모달리티 및 API 추가 시 변경해야 할 것들
* *!!* 가 달려 있는 항목은 API별 한 번씩 작업해야 함.
* *!!* 가 달려 있지 않는 항목은 모달리티별 한 번씩 작업해야 함.

* `main.py`
  1. `parse_args()`의 **api(29-34 lines)** 에서 choice에 사용할 API 이름 추가 *!!*
  2. `parse_args()`의 **modality(24-28 lines)** 에서 choice에 사용할 모달리티 추가
  3. **Computing histogram 다음 부분** 에서 `visualize()`의 사용 여부 결정
  4. `visualize()`에서 다중 샘플 시각화 방법 정의

* `dpsda/data_logger.py`
  1. `log_samples()`의 **(5~ lines)** 에서 모달리티에 따른 샘플 저장 방법 정의

* `dpsda/dataset.py`
  1. 모달리티에 따른 확장자 변수 **EXTENSIONS(7~ lines)** 정의
  2. 모달리티에 따른 **Dataset 클래스** 정의
  
* `dpsda/data_loader.py`
  1. `load_data()`의 **(13~ lines)** 에서 모달리티에 따른 dataset 정의
  2. `load_data()`의 **for문 아래로 첫 번째 if** 에서 모달리티에 따른 tensor 처리 과정 정의
   
* `dpsda/feature_extractor.py`
  1. `extract_features()`의 **(27~ lines)** 에서 모달리티에 따른 feature extract 방법 정의

* `apis/` 
  1. 사용할 API 클래스 정의 *!!*
     1. 기존의 정의된 파일을 기반으로 `__init__()`에서 필요한 변수 정의
        * `__init__()`이 인자로 받는 모든 변수는 `command_line_parser()`에서 정의된 argument여야 함.
     2. `random_sampling()` 동작 방법 정의
         * 내부 for문 위주로 변경하면 됨.
     3. `variation()` 동작 방법 정의
        * variation degree가 정의되지 않는 API의 경우 DPSDA 논문에 따라 variation degree를 정수로 설정하고 그 크기만큼 반복 호출하도록 정의하였음. 추후 논의 필요.

* `apis/__init__.py`
  1. `get_api_class_from_name`에서 사용할 API 클래스 개체를 반환해주는 조건문 추가 *!!* - `main.py`의 i번 항목에서 추가한 API 이름과 동일해야 함

## Sampling/variation 도중에 중단된 실험 재개하기
1. `--save_samples_live` 추가하여 중간 결과물을 저장하도록 설정
   * `--result_folder`로 지정한 경로에 `initial_{iteration}_samples.npz`와 `variation_{variation}_{candidate}_samples.npz`, `sub_variation_{variation}_{iteration}`로 저장됨
2. `--live_loading_target`으로 불러올 중간 결과물의 경로 지정
   * 불러올 중간 결과물이 없다면 필요 없음
3. `--save_samples_live_freq`로 저장 주기(배치 단위) 설정
