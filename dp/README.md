# DPSDA
[DPSDA Github](https://github.com/microsoft/DPSDA) 참고

## 2024.01.24. Updates
### Parameters
#### Ours
* `direct_variate`: 1226 그거 적용 여부, *default=False*
* `sample_weight`: sample-based variation에 대한 가중치, *default=1.0*
* `demonstration`: variation 시 demostration으로 보여줄 샘플의 개수, *default=0*
* `adaptive_variation_degree`: 샘플마다 variation degree를 다르게 설정할지 여부, *default=False*



#### Schedulers
* 구현된 종류: constant, linear, step, exponential
* 스케줄러별 상세 argument 목록은 `/dpsda/schedulers.py`의 각 스케줄러 클래스의 `command_line_parser()` 함수에서 확인 가능
  * 만약 사용하고자 하는 스케줄러의 `command_line_parser()`에 `min`이라는 변수가 있다면 `[scheduler type]_scheduler_min`으로 설정
  * 예: `degree_scheduler_min`, `weight_scheduler_min`
1. **Degree scheduler**: 
  * `use_degree_scheduler`: 스케줄러 사용 여부, *default=True*
  * `degree_scheduler`: 사용할 스케줄러 이름, *default=constant*
2. **Weight scheduler**
  * `use_weight_scheduler`: 스케줄러 사용 여부, *default=True*
  * `weight_scheduler`: 사용할 스케줄러 이름, *default=constant*





#### Privacy
* `dp`: DP 적용 여부, *default=True*
* `delta`: epsilon DP 완화 정도 *default=0.0*
* `epsilon_delta_dp`: 1/N_syn 값으로 delta 값 설정 여부, True일 경우 `delta` 값 무시, *default=True*






#### General
* `random_seed`: 난수 조절, *default=2024*
* `num_samples`: 몇 개의 샘플을 생성할지, *default=0*
* `T`: 몇 번 동안 반복할지 *default=0*
  * `num_samples`와 `T`는 동시에 설정되어야 함.
  * 둘 중 하나만 할 경우 이전에 `100,100,~` 했던 것처럼 설정한 거로 보기 때문에 예상치 못한 결과 발생 가능
* `use_public_data`: public seed population 여부, *default=False*
* `public_data_folder`: seed population으로 사용할 public data의 경로, `use_public_data`가 False일 경우 무시됨.


#### Wandb
***어떤 실험을 돌리든 자동으로 wandb에 기록됨***
* `wandb_log_notes`: 실험을 간략하게 소개하는 문구 지정, 추후 wandb 사이트에서 수정 가능
* `wandb_log_tags`: 실험을 분류하기 위한 태그, 추후 wandb 사이트에서 수정 가능 
* `wandb_log_dir`: wandb log를 저장할 경로, *default=/mnt/cubigate*
* `wandb_resume_id`: 재개할 실험의 wandb id
  * wandb 페이지 상에서 이어서 하고자 하는 실험의 상세 페이지(overview)를 보면 **Run path**라는 항목이 있음
  * Run path에서 **cubig_ai/AZOO/** 이후의 값이 해당 실험의 id임


---
## Parameter setting examples
### Initial population
* **random**: 특별히 신경써야 할 인자 없음
* **public**:
  ```
  --use_public_data true --public_data_folder [PUBLIC_DATA_FOLDER]
  ```

### DPSDA
*  privacy나 general 관련 파라미터와 degree scheduler는 DPSDA에도 적용 가능
  ```
  --direct_variate false --use_weight_scheduler false --adaptive_variation_degree false
  ```


### Demonstration
* **적용X**:
 ```
 --use_weight_scheduler false
 ```
* **적용O**:
 1. 스케줄러 사용
 ```
 --weight_scheduler linear --weight_scheduler_base 1.0 --weight_scheduler_min 0.9 --demonstration 3
 ```
 2. 스케줄러 미사용
   * constant scheduler와 동일한 결과
 ```
 --use_weight_scheduler false --sample_weight 0.9 --demonstration 3
 ```



### Resumption
* `data_checkpoint_step`: `_samples.npz` 파일의 경로
* `count_checkpoint_path`: `count.npz` 파일의 경로
  * DPSDA는 해당 파일이 없는 실험도 재개할 수 있지만 우리 거는 필수
* `wandb_resume_id`: wandb 상에서도 별개의 실험으로 기록이 분리되지 않으려면 필요함.
  * 설정 안 해도 오류는 안 나나 로그를 보기 불편할 수 있음.
  * 참고: **result_folder 안에 wandb run name으로 실험마다 따로 결과물 저장함**




### All together

<details>
<summary> Cookie 예시 </summary>

```
python main.py \
--device 2 \
--api_device 2 \
--count_threshold 2.0 \
--feature_extractor inception_v3 \
--fid_model_name inception_v3 \
--fid_dataset_name cookie_dp \
--num_candidate 8 \
--private_sample_size 512 \
--sample_size 512x512 \
--data_folder /mnt/cubigate/data/cookie \
--num_samples 100 \
--T 17 \
--num_fid_samples 100 \
--num_private_samples 100 \
--initial_prompt "a realistic photo of white ragdoll cat" \
--make_fid_stats true \
--result_folder /mnt/cubigate/minsy/result/cookie \
--tmp_folder /tmp/ragdoll_result_pet_test/minsy \
--api stable_diffusion \
--random_sampling_checkpoint digiplay/AbsoluteReality_v1.8.1 \
--random_sampling_guidance_scale 7.5 \
--random_sampling_num_inference_steps 20 \
--random_sampling_batch_size 8 \
--variation_checkpoint digiplay/AbsoluteReality_v1.8.1 \
--variation_guidance_scale 7.5 \
--variation_num_inference_steps 20 \
--variation_batch_size 1  \
--modality image \
--direct_variate true \
--dp true \
--epsilon 2.0 \
--use_degree_scheduler true \
--degree_scheduler linear \
--degree_scheduler_base 1.0 \
--degree_scheduler_min 0.2 \
--use_weight_scheduler true \
--weight_scheduler linear \
--weight_scheduler_base 1.0 \
--weight_scheduler_min 0.8 \
--demonstration 3 \
--adaptive_variation_degree true \
--data_checkpoint_path "/mnt/cubigate/minsy/result/cookie/fluent-lion-1570/1/_samples.npz" \
--data_checkpoint_step 1 \
--count_checkpoint_path "/mnt/cubigate/minsy/result/cookie/fluent-lion-1570/1/count.npz" \
--wandb_resume_id tyk5q804 \
--wandb_log_tags cookie
```
</details>

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
