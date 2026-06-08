from sentence_transformers.cross_encoder import CrossEncoder
# Chạy thêm test này
ce_pretrained = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)
scores_pre = ce_pretrained.predict([
    ("AI in healthcare", "Doctors use AI to diagnose cancer earlier"),
    ("AI in healthcare", "The football match ended 2-1 on Saturday"),
])
print("Pretrained:", scores_pre)

ce_finetuned = CrossEncoder("models/cross_encoder", max_length=256)
scores_ft = ce_finetuned.predict([
    ("AI in healthcare", "Doctors use AI to diagnose cancer earlier"),
    ("AI in healthcare", "The football match ended 2-1 on Saturday"),
])
print("Finetuned:", scores_ft)