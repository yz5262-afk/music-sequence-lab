# Source and publication notes

This repository was prepared from the owner's local project: **Applied Machine Learning / edited RNN-for-music notebook**.

Coursework-derived project. The source explicitly credits the MIT Deep Learning Lab 1 music example; that credit is preserved. Existing model/training/sampling code is retained in a cleaned notebook. Exaggerated marketing headings, assignment prose, execution outputs and generated audio are omitted. The lightweight sampler is exported from that notebook.

The public package removes notebook execution outputs, machine-specific paths,
environment files, personal documents and unrelated data. Source notebooks are
presented as research or coursework, with their existing methods and limitations.
Packaging adds documentation and small examples; it does not establish novel
algorithm authorship or a measured production deployment.

No new open-source license is granted by this packaging. Existing third-party
attributions and applicable upstream licenses continue to apply.

The public sampler also corrects the sign of the repetition penalty: repeated negative logits are multiplied by the penalty, rather than divided by it.
