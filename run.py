
from models import *
from train import *

if __name__ == '__main__':
    # 데이터셋 생성
    # dataset = ECGDataset(ecg_data, k_values)
    #
    # # DataLoader로 배치 처리
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # # 샘플 데이터 생성
    # k_values = np.random.uniform(1.5, 7.5, size=100)  # K+ 농도 샘플 생성
    # encoded_labels = [encode_k_concentration(encode_k_class(k)) for k in k_values]  # 클래스 라벨로 변환
    #
    # # PyTorch Tensor로 변환
    # labels = torch.tensor(encoded_labels, dtype=torch.long)  # 분류 문제이므로 long 타입 사용
    # print(labels)
    model = ECG12Net2()
    example_input = torch.randn(1, 12, 1024)
    emp_input = torch.randn(1, 8)

    output = model(example_input,emp_input)
    print(output)

