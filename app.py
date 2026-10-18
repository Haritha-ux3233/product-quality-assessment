
import gradio as gr
import pandas as pd
import joblib

# Load the trained Random Forest model
rf_model = joblib.load("rf_model.pkl")

FEATURES = [
    "ProductionVolume",
    "ProductionCost",
    "SupplierQuality",
    "DeliveryDelay",
    "DefectRate",
    "QualityScore",
    "MaintenanceHours",
    "DowntimePercentage",
    "EquipmentUtilization",
    "WorkerProductivity",
    "SafetyIncidents",
    "EnergyConsumption",
]


def assess_quality(
    volume, cost, supplier, delay, defect, quality,
    maintenance, downtime, utilization, productivity,
    safety, energy
):
    values = [
        volume, cost, supplier, delay, defect, quality,
        maintenance, downtime, utilization, productivity,
        safety, energy
    ]

    if any(v is None for v in values):
        return "<h3>Please fill in all fields.</h3>"

    if any(v < 0 for v in values):
        return "<h3>Values cannot be negative.</h3>"

    percentage_fields = [
        supplier, defect, quality, downtime,
        utilization, productivity
    ]
    if any(v > 100 for v in percentage_fields):
        return "<h3>Score and percentage fields must be between 0 and 100.</h3>"

    input_data = pd.DataFrame([values], columns=FEATURES)

    prediction = int(rf_model.predict(input_data)[0])
    probabilities = rf_model.predict_proba(input_data)[0]
    classes = list(rf_model.classes_)

    defect_probability = (
        float(probabilities[classes.index(1)])
        if 1 in classes else 0.0
    )

    status = "DEFECTIVE" if prediction == 1 else "GOOD"

    if defect_probability < 0.30:
        risk = "LOW RISK"
        color = "#15803d"
        recommendation = "Continue routine quality monitoring."
    elif defect_probability < 0.60:
        risk = "MEDIUM RISK"
        color = "#b45309"
        recommendation = "Perform additional quality inspections."
    else:
        risk = "HIGH RISK"
        color = "#dc2626"
        recommendation = (
            "Inspect the product and investigate the production process."
        )

    probability_percent = defect_probability * 100

    return f"""
    <div style="background:white; padding:28px; border-radius:18px;
                border:1px solid #e2e8f0; color:#0f172a;">
        <h2 style="color:{color}; margin-top:0;">{status}</h2>
        <p><b>Estimated defect probability</b></p>
        <h1 style="color:{color}; font-size:36px;">
            {probability_percent:.2f}%
        </h1>
        <div style="background:#e2e8f0; height:14px; border-radius:10px;">
            <div style="width:{probability_percent:.2f}%;
                        height:14px; background:{color};
                        border-radius:10px;"></div>
        </div>
        <h3 style="color:{color};">{risk}</h3>
        <p><b>Recommendation:</b> {recommendation}</p>
        <p style="color:#64748b; font-size:13px;">
            Predictions depend on the trained model and the quality of its
            training data. Verify results against actual inspections.
        </p>
    </div>
    """


css = """
.gradio-container {
    max-width: 100% !important;
    background: #f1f5f9 !important;
}
.header {
    background: linear-gradient(135deg, #0f172a, #1d4ed8);
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 20px;
    color: white !important;
}
.header h1, .header p {
    color: white !important;
}
.input-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
}
"""


with gr.Blocks(
    title="Product Quality & Defect Risk Assessment",
    fill_width=True
) as demo:

    gr.HTML("""
    <div class="header">
        <h1>Product Quality &amp; Defect Risk Assessment</h1>
        <p>AI-Powered Manufacturing Quality Control System</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(elem_classes=["input-card"]):
            gr.Markdown("### Production Parameters")
            volume = gr.Number(label="Production Volume", value=750)
            cost = gr.Number(label="Production Cost", value=15000)
            supplier = gr.Number(label="Supplier Quality (0–100)", value=88)
            delay = gr.Number(label="Delivery Delay (Days)", value=3)
            defect = gr.Number(label="Defect Rate (%)", value=3.8)
            quality = gr.Number(label="Quality Score (0–100)", value=72)

        with gr.Column(elem_classes=["input-card"]):
            gr.Markdown("### Equipment & Workforce")
            maintenance = gr.Number(label="Maintenance Hours", value=70)
            downtime = gr.Number(label="Downtime (%)", value=12)
            utilization = gr.Number(
                label="Equipment Utilization (%)", value=68
            )
            productivity = gr.Number(
                label="Worker Productivity (%)", value=75
            )
            safety = gr.Number(label="Safety Incidents", value=4)
            energy = gr.Number(label="Energy Consumption", value=700)

    predict_btn = gr.Button(
        "ASSESS PRODUCT QUALITY",
        variant="primary"
    )

    result = gr.HTML("""
    <div style="background:white; padding:25px; border-radius:16px;">
        Change the input values and click Assess Product Quality.
        Your assessment will appear here.
    </div>
    """)

    inputs = [
        volume, cost, supplier, delay, defect, quality,
        maintenance, downtime, utilization, productivity,
        safety, energy
    ]

    predict_btn.click(
        fn=assess_quality,
        inputs=inputs,
        outputs=result
    )

demo.launch(css=css)
