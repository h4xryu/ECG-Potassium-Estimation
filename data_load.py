from utils.dataset.Severance import *
from utils.one_cycle_ecg import *
import torch
from scipy.signal import resample

class MultiLeadECGDataset(Dataset):
    def __init__(self, root='./dataset'):
        # aVF-aVL-aVR-I-II-III-V1-~-V6 순서로 정렬
        self.lead_order = ['I', 'II', 'III', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6','aVF', 'aVL', 'aVR' ]

        self.lead_info = {i: lead for i, lead in enumerate(self.lead_order)}

        self.ecg_data = []
        self.k_levels = None
        self.patient_ids = None

        # 임시 저장소
        lead_data = {}  # 각 리드별 데이터 저장
        patient_data = {}  # 각 리드별 환자 ID 저장
        k_level_data = {}  # 각 리드별 칼륨 수치 저장

        # 각 리드별로 데이터 로드
        for i, lead in enumerate(self.lead_order):
            if lead == 'II' or lead == 'v5':
                # A 클래스 데이터 로드
                ecg_a, pat_a, spl_a = get_ecg_from_lead(root=root, cls='A', lead=lead)

                # B 클래스 데이터 로드
                ecg_b, pat_b, spl_b = get_ecg_from_lead(root=root, cls='B', lead=lead)

                # 데이터 결합
                ecg = ecg_a + ecg_b
                pat = pat_a + pat_b
                spl = spl_a + spl_b

                ecg, spl, pat = one_cycle(ecg, spl, pat)

                # 리드별 데이터 저장
                lead_data[lead] = ecg
                patient_data[lead] = pat
                k_level_data[lead] = spl

        # 모든 리드에 공통으로 존재하는 환자 ID 찾기
        self.selected_lead_order = ['II','v5']
        self.lead_info = {i: lead for i, lead in enumerate(self.selected_lead_order)}
        common_patients = set(patient_data[self.selected_lead_order[0]])
        for lead in self.selected_lead_order[1:]:
            common_patients = common_patients.intersection(set(patient_data[lead]))

        # 공통 환자 ID 목록
        self.patient_ids = sorted(list(common_patients))

        # 공통 환자의 데이터만 추출 및 리샘플링
        all_lead_data = []
        for lead in self.selected_lead_order:
            lead_ecg = []
            lead_indices = [patient_data[lead].index(pid) for pid in self.patient_ids]

            for idx in lead_indices:
                current_signal = lead_data[lead][idx]
                resampled_signal = self.resample_signal(current_signal, 512)
                lead_ecg.append(resampled_signal)

            # 리드별 데이터를 numpy 배열로 변환
            lead_ecg = np.array(lead_ecg)
            all_lead_data.append(lead_ecg)

        # 모든 리드 데이터를 하나의 numpy 배열로 결합
        all_lead_data = np.array(all_lead_data)  # shape: (12, N, 512)
        all_lead_data = np.transpose(all_lead_data, (1, 0, 2))  # shape: (N, 12, 512)

        # numpy 배열을 torch tensor로 변환
        self.ecg_data = torch.FloatTensor(all_lead_data)

        # 칼륨 수치도 공통 환자의 것만 추출
        first_lead = self.selected_lead_order[0]
        first_lead_indices = [patient_data[first_lead].index(pid) for pid in self.patient_ids]
        self.k_levels = torch.FloatTensor([k_level_data[first_lead][idx] for idx in first_lead_indices])

        print(f"Total number of samples after alignment: {len(self.patient_ids)}")
        print(f"ECG data shape: {self.ecg_data.shape}")

    def resample_signal(self, signal, target_length):
        """리샘플링 함수: 신호를 목표 길이로 조정"""
        if len(signal) == target_length:
            return signal
        elif len(signal) == 5000:
            return signal[100:target_length+100]
        return resample(signal, target_length)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        return self.ecg_data[idx], self.k_levels[idx]
