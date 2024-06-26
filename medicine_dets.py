medicine_to_include = [
    "Albendazole",
    "Amoxycillin",
    "Ascabiol",
    "Azithromycin",
    "Beclo + Clotri + Genta cream",
    "Becosule",
    "Calcium +",
    "Cetrizine",
    "Ciplox eye drops",
    "Ciprofloxacin",
    "Clotrimazole",
    "Cough syrup",
    "Diclofenac gel",
    "Dicyclomine + Mefenamic acid",
    "dicyclomine (10mg) + mefenamic acid (250mg)",
    "norfloxacin (200mg) + tinidazole (300mg)",
    "norflox",
    "omez",
    "vitamin-c chewable 500mg tablet",
    "unizyme",
    "becousles",
    "omez",
    "Doxycycline",
    "Dulcoflex",
    "Herbal cough",
    "Iron",
    "Liver tonic",
    "Miconazole nitrate",
    "Multivitamin",
    "Norfloxacin + Tinidazole",
    "Omeprazole",
    "Ondansetron",
    "ORS",
    "Paracetamol",
    "Aceclo + PCM",
    "Fluconazole",
    "Neurobion forte",
    "Unienzyme",
    "Tetmosol soap",
    "Vitamin C",
    "Prochlorperazine",
    "PCM",
    "Limcee",
    "Ivermectin",
    "Benedryil syrup",
    "Amoxy",
    "Metronidazole",
    "Miconazole Nitrate cream",
    "Pediatric Multivitamin Syp",
    "Prochlorperazine stemetil md",
    "Aceclo + PCM (SOS)",
    "Fluconazole",
    "Tetmasol Soap",
    "Vitamin C Chewable",
    "Waxonil Ear Drop",
    "calcium 250 mg+vit d3",
]

medicine_to_include_lower = [name.lower() for name in medicine_to_include]
mappings = {
    "unienzyme": "unizyme",
    "becosule": "becousles",
    "tetmasol soap": "tetmosol soap",
    "vitamin c": "vitamin-c chewable 500mg tablet",
    "amoxycillin": "amoxy",
    "norflox": "norfloxacin (200mg) + tinidazole (300mg)",
    "calcium +": "calcium 250 mg+vit d3",
}

min_months_required = 16
import pandas as pd
def has_sufficient_data(group):
    unique_months = pd.to_datetime(group["Date"]).dt.to_period("M").nunique()
      # print(unique_months)
    return unique_months >= min_months_required
