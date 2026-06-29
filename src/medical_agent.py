try:
    import torch
    from transformers import pipeline
except ImportError:
    torch = None
    pipeline = None

class MedicalLLMAgent:
    """
    Autonomous Chief Medical Officer Agent.
    Synthesizes numerical outputs from XGBoost and Neural ODEs into a
    natural-language clinical recommendation for 15-PGDH inhibitor protocols.
    """
    def __init__(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        # For a truly local lightweight deployment on average hardware, 
        # using a small instruction-tuned model. In production, this might be Llama-3-8B.
        print(f"Loading local LLM Agent ({model_id})...")
        try:
            if pipeline is None:
                print("Transformers pipeline not available. Running in simulation mode.")
                self.generator = None
            else:
                self.generator = pipeline(
                    "text-generation", 
                    model=model_id, 
                    device_map="auto"
                )
        except Exception as e:
            print(f"Error loading LLM {model_id}: {e}")
            self.generator = None

    def generate_cmo_summary(self, telemetry_data: dict, xgboost_risk_score: float, ode_trajectory: dict) -> str:
        """
        Generates the Chief Medical Officer Executive Summary.
        """
        if self.generator is None:
            return "ERROR: LLM Agent Pipeline not initialized. Check model weights and environment."
            
        # Parse inputs
        heart_rate = telemetry_data.get("heart_rate", "N/A")
        bone_density = telemetry_data.get("bone_density_t_score", "N/A")
        
        final_matrix_integrity = ode_trajectory.get("final_matrix_integrity", 0.0)
        recommended_drug_dose = ode_trajectory.get("recommended_drug_dose", 0.0)
        
        # Construct the context prompt
        system_prompt = (
            "You are the autonomous Chief Medical Officer for a deep space mission. "
            "Your role is to analyze biometric telemetry, AI risk scores, and mathematical models to provide clinical recommendations. "
            "Write a concise, professional executive summary recommending a 15-PGDH inhibitor dosage adjustment."
        )
        
        user_prompt = (
            f"Astronaut Telemetry: Heart Rate: {heart_rate} bpm, Bone Density T-Score: {bone_density}.\n"
            f"XGBoost Mission Failure Risk Score: {xgboost_risk_score:.2f} (0=Low, 1=High).\n"
            f"Neural ODE 6-Month Projection - Final Cartilage Matrix Integrity: {final_matrix_integrity:.2f} (0=Destroyed, 1=Healthy).\n"
            f"Neural ODE Recommended 15-PGDH Inhibitor Dose: {recommended_drug_dose:.2f} mg.\n"
            "Based on these numbers, provide a short, biochemical reasoning for the recommended dosage and the overall risk assessment."
        )
        
        # For TinyLlama chat format
        prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>\n"
        
        try:
            print("Generating CMO summary...")
            outputs = self.generator(prompt, max_new_tokens=250, do_sample=True, temperature=0.7, top_p=0.9)
            generated_text = outputs[0]['generated_text']
            
            # Extract just the assistant's response
            summary = generated_text.split("<|assistant|>\n")[-1].strip()
            return summary
            
        except Exception as e:
            print(f"Inference error: {e}")
            return "ERROR: Failed to generate summary during inference."

if __name__ == "__main__":
    # Test initialization
    # agent = MedicalLLMAgent()
    pass
