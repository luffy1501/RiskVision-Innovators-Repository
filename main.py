import pandas as pd
from models.autoencoder import train_autoencoder, detect_anomalies
from models.gan import train_gan
from models.vae import build_vae, train_vae
from chatbot.chatbot import setup_openai, get_mitigation_strategy
from compliance.compliance import check_compliance


def load_data():
    market_data = pd.read_csv('data/market_data.csv')
    customer_transactions = pd.read_csv('data/customer_transactions.csv')
    operational_logs = pd.read_csv('data/operational_logs.csv')
    combined_data = pd.concat([market_data, customer_transactions, operational_logs], axis=1)
    return combined_data


def main():
    # Load data
    combined_data = load_data()

    # Anomaly Detection
    autoencoder = train_autoencoder(combined_data.values)
    anomalies = detect_anomalies(autoencoder, combined_data.values)
    print("Anomalies detected:\n", combined_data[anomalies])

    # GAN for Credit Risk
    generator, discriminator = train_gan(combined_data.values)

    # VAE for Market Risk
    vae = build_vae(combined_data.shape[1])
    vae = train_vae(vae, combined_data.values)

    # Chatbot for Mitigation Strategy
    setup_openai('your-openai-api-key')
    risk_type = "credit"
    risk_details = "High probability of default for customer segment X."
    mitigation_strategy = get_mitigation_strategy(risk_type, risk_details)
    print("Mitigation Strategy:\n", mitigation_strategy)

    # Compliance Monitoring
    regulatory_text = "According to the new regulation ABC123, all transactions must be logged."
    compliance_issues = check_compliance(regulatory_text)
    print("Compliance Issues:\n", compliance_issues)


if __name__ == "__main__":
    main()
