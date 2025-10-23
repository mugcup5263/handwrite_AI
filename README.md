# Handwriting Recognition App

손글씨 숫자 인식을 위한 Kivy 기반 Android 앱입니다.

## 기능
- 손글씨 숫자 인식 (0-9)
- 신경망 구조 시각화
- 실시간 예측 결과 표시

## 파일 구조
- `handwriting_gui.py` - 메인 앱
- `buildozer.spec` - APK 빌드 설정
- `mnist_cnn.h5` - 훈련된 모델 파일

## APK 빌드
GitHub Actions를 통해 자동으로 APK가 빌드됩니다.

1. 코드를 main 브랜치에 푸시
2. Actions 탭에서 빌드 진행 상황 확인
3. 완료 후 Artifacts에서 APK 다운로드

## 사용법
1. APK 설치
2. 앱 실행
3. 화면에 숫자 그리기
4. "예측" 버튼 클릭
5. 결과 확인
