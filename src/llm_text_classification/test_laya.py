import laya

# Preload the model (downloads weights to local cache on first run)
print("Loading Laya model...")
agent = laya.load("convaiinnovations/laya")
print("Model loaded. Running test inference...\n")

# Run test decision pass
result = agent.system_one(
    state={
        "criterion": "Documented history of type 2 diabetes mellitus with HbA1c > 8.5% within 90 days prior to screening."
    },
    questions={
        "eligibility_domain": {
            "type": "choice",
            "instructions": "Which clinical trial eligibility domain does this criterion describe?",
            "criteria": {
                "condition": "A disease, diagnosis, sign, or symptom",
                "measurement": "A laboratory test value, biomarker, or vital sign",
                "drug": "A pharmaceutical agent, medication, or therapy",
                "procedure": "A surgical, diagnostic, or therapeutic procedure"
            }
        },
        "has_lab_threshold": {
            "type": "noul",
            "instructions": "Does this criterion specify an explicit numerical laboratory measurement or threshold?"
        }
    }
)

# Display results
print("--- INFERENCE RESULTS ---")
print("Selected Choice :", result["answers"]["eligibility_domain"]["choice"])
print("Probabilities   :", result["answers"]["eligibility_domain"]["probabilities"])
print("Has Lab Value   :", result["answers"]["has_lab_threshold"]["noul"], "(P(true))")
print("Input Tokens    :", result["usage"]["input_tokens"])