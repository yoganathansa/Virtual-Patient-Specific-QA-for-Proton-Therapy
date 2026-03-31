# Virtual-Patient-Specific-QA-for-Proton-Therapy
This project presents a virtual quality assurance (QA) framework for proton therapy that predicts measurement fluence from Treatment Planning System (TPS) data. The goal is to reduce reliance on time-consuming physical QA measurements by leveraging computational modeling to verify treatment delivery accuracy.

Patient-specific quality assurance (PSQA) is a critical step in proton therapy to ensure that planned dose distributions are accurately delivered. Traditional QA methods rely on physical measurements, which can be resource-intensive and time-consuming.

This project introduces a virtual PSQA pipeline that predicts measured fluence maps directly from TPS outputs, allowing for rapid and scalable QA verification.

Input
TPS-generated fluence maps
Output
Predicted measurement fluence maps
Approach
Data preprocessing and normalization of TPS and measured fluence
Supervised learning model (e.g., CNN / regression-based architecture)
Training on paired TPS–measurement datasets
Evaluation using gamma analysis and error metrics

⚠️ Disclaimer

This project is intended for research purposes only and is not approved for clinical use without proper validation and regulatory compliance.

## 👤 Author

**SA Yoganathan**  
Medical Physicist  
Saint John Regional Hospital  
Saint John, NB, Canada  

---

### 📬 Contact
For questions, collaborations, or research inquiries, please open an issue or contact via GitHub.
