# Final Project for Deep Learning OMSCS

Read the full paper here: [report]report.pdf

## Abstract
In this paper, we address the challenges of automatic
music tagging, a critical task within the field of Music In-
formation Retrieval (MIR) that involves classifying audio
tracks into multiple labels such as genre and mood. We im-
prove the robustness of music tagging models against real-
world audio variability and propose a new a Vision Trans-
former (ViT) architecture. We leverage the MagnaTagATune
dataset, comprising approximately 25,000 music clips, and
apply stochastic augmentations including noise addition,
time stretching, and pitch shifting during training. Our ex-
periments reveal that while traditional models benefit from
these augmentations, the ViT model exhibits unique behav-
iors, suggesting potential overfitting challenges. Despite
these issues, the ViT architecture achieves tagging perfor-
mance comparable to state-of-the-art models, indicating its
promise for future research in music classification. Our
findings suggest that effective data augmentation can signif-
icantly enhance model resilience, paving the way for more
robust applications in MIR.
