import pandas as pd
import numpy as np
from faker import Faker
import random

# Initialize Faker with Indonesian locale
fake = Faker('id_ID')

# Set seed to ensure reproducible random results every time the script is run
np.random.seed(42)
random.seed(42)

# CHANGED: Number of students reduced to 200
NUM_STUDENTS = 200
SUBJECTS = ['MTK', 'B_Indo', 'B_Inggris', 'IPA', 'IPS']


def generate_student_data():
    data = []

    for i in range(NUM_STUDENTS):
        # 1. Basic Student Profile
        student_id = f"SIS-{str(i + 1).zfill(4)}"
        nama = fake.name()
        tgl_lahir = fake.date_of_birth(minimum_age=14, maximum_age=16).strftime('%Y-%m-%d')
        asal_sekolah = f"SMPN {random.randint(1, 10)} {fake.city()}"

        # 2. CHANGED: Tightened base intelligence to target the 80-87 sweet spot.
        # Mean is set to 82, but with a small standard deviation (3) so grades stay compressed.
        base_intel = np.random.normal(82, 3)
        base_intel = np.clip(base_intel, 70, 88)  # Cap base at 88 so 90+ is incredibly hard to reach

        # 3. CHANGED: Reduced the trend intensity so grades don't inflate wildly over 6 semesters
        trend = np.random.choice([-0.8, -0.3, 0, 0.2, 0.5, 0.8],
                                 p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05])

        student_record = {
            'id_siswa': student_id,
            'nama': nama,
            'tanggal_lahir': tgl_lahir,
            'asal_sekolah': asal_sekolah
        }

        # 4. Generate Grades per Subject per Semester
        total_semester_6 = 0
        for subj in SUBJECTS:
            # CHANGED: Reduced subject bias variance to keep scores stable across subjects
            subj_bias = np.random.normal(0, 2)
            subj_base = base_intel + subj_bias

            for sem in range(1, 7):
                # CHANGED: Reduced random noise to prevent accidental spikes up to 95-100
                grade = subj_base + (trend * sem) + np.random.normal(0, 1.5)
                grade = np.clip(grade, 40, 100)

                student_record[f'{subj}_Sem{sem}'] = round(grade, 1)

                if sem == 6:
                    total_semester_6 += grade

        # 5. Eligibility Determination
        rata_rata_akhir = total_semester_6 / len(SUBJECTS)
        student_record['rata_rata_akhir'] = round(rata_rata_akhir, 1)

        # Retained original eligibility rule (average >= 80 and not failing trend)
        if rata_rata_akhir >= 80 and trend >= -0.5:
            student_record['status_eligible'] = 1
        else:
            student_record['status_eligible'] = 0

        data.append(student_record)

    return pd.DataFrame(data)


if __name__ == "__main__":
    print("Starting data generation for 200 students...")
    df = generate_student_data()

    # Save to a CSV file
    output_filename = "dataset_siswa_test.csv"
    df.to_csv(output_filename, index=False)

    print(f"✅ Done! Data successfully saved to file: {output_filename}")

    # Display a brief summary
    print("\nData Summary:")
    print(df['status_eligible'].value_counts().rename({1: 'Eligible', 0: 'Not Eligible'}))
    print(f"Overall Dataset Average Score: {round(df['rata_rata_akhir'].mean(), 2)}")
    print(f"Highest Score in Dataset: {df['rata_rata_akhir'].max()}")

    print("\nFirst 5 Rows of Data:")
    print(df.head())