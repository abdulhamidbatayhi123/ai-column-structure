**AI-use policies for authors: IOP, Springer Nature, Elsevier and TMLR, checked against the author's disclosure wording (all pages read live 2026-11-16)**

I read every page in full with curl. Nothing on iopscience.iop.org was needed, so the bot check did not get in the way. Raw copies are in `C:\Users\ABDULH~1\AppData\Local\Temp\claude\c6eff6e074d30b33\4add04dd-e721-4eee-b705-23ef12c2b27a\scratchpad\` (`iop_ai.html.txt`, `p7461.html.txt`, `p7454.html.txt`, `ethics.html.txt`, `natai.html.txt`, `sn_*.txt`, `els.html.txt`, `tmlr_*.txt`). I changed no project files.

## 1. IOP Publishing (MLST)

Sources:
- **Authors page:** publishingsupport.iopscience.iop.org/questions/generative-ai-tools/. Its page metadata gives dateModified 2026-10-15 and datePublished 2024-03-05.
- **Ethical Policy for Journals:** /ethical-policy-journals/ (modified 2026-09-04). It contains the same authors text word for word, and MLST is on the list of journals it covers.
- **Reviewers:** /questions/generative-ai-for-reviewers/ (modified 2026-10-01) and /ethics-for-reviewers/ (modified 2026-07-22).

**Allowed:**
- **Research process and code:** "code generation, data analysis and anomaly detection … provided their use is disclosed where required and any outputs are appropriately validated by the authors."
- **Writing:**
  - "Edit human-written text."
  - "Generate text – authors must ensure that they have critically revised any AI-generated text".
- **Figures:**
  - "Generate figures based on existing data … including … AI-assisted coding"
  - but this must not "alter, distort or misrepresent the underlying data".
  - Explanatory images must be "clearly distinguishable from research data".
- **Literature:** AI may "locate, retrieve, organise and summarise" publications; the synthesis must be the author's own.
- **Replies to reviewers:** "only to improve the language of their responses to peer review reports."

**Prohibited:**
- Fabricating or altering data.
- **Reference lists:** "Generate or modify reference lists". AI may "support literature discovery, citation management, reference formatting and reference checking, but every cited source must be verified by the authors against the original source".
- **Reviewer reports:** "Upload reports from reviewers to generative AI tools".
- **Responses to reviewers:** "Generate responses to reviewers … Authors must personally evaluate reviewer comments".
- Hidden prompts, which mean "immediate rejection".
- Citing a non-existent source is "strong evidence of irresponsible AI usage". It usually leads to rejection, and after publication possibly retraction.

**Disclosure:**
- **Where:** "a separate AI disclosure statement as part of the Acknowledgement section".
- **What it must say:** "identify the tool or model used, describe how it was used and explain the extent to which the authors verified, validated and critically reviewed the AI-assisted outputs."
- **Methods too:** use in the research process (code counts) must be "additionally described in detail in the Methods".
- **Exemption:** routine spelling, grammar and language editing needs no statement.
- **Possible extra request:** prompts, logs and outputs as supplementary data.
- **Template, exact:** "During the preparation of this manuscript, the authors used [Tool Name, Version] to support [describe use]. The authors reviewed, validated and critically assessed all AI-assisted outputs and take full responsibility for the content of the manuscript."

**Reviewers:** AI may be used "only to assist with improving the written quality" of a report. Reviewers may not put manuscript material into AI tools or use AI to "generate references or citations".

This settles the disagreement the playbook critic raised (item 2): both reports quoted part of one two-sentence template. The "reviewed, validated and critically assessed" sentence is real and current.

## 2. Springer Nature

Sources: nature.com/nature-portfolio/editorial-policies/ai, springernature.com/gp/policies/editorial-policies/ai-manuscript-preparation and …/using-ai-in-research. None of these pages shows a date.

SN now grades AI use by risk level: Green, Amber and Red.

**Green / Amber (allowed):**
- Language editing.
- "drafting sections based on author input".
- "reference management support".
- "code scaffolding".
- "supporting statistical modelling".

Declaration for these: "Clearly describe use and confirm author accountability". For analysis use: "Provide clear methodological description and validation steps".

**Red (not permitted):**
- "Writing results or discussion independently".
- "Fabricating data, citations or results".
- "Delegating peer review to an LLM".
- Listing AI as an author.

**Confidentiality:** this applies to every role, authors included: "Manuscripts, peer review reports, and sensitive data must not be shared with unsecured or public AI systems". Pasting reviewer reports into a public chatbot is therefore also a problem at SN.

**Figures:** allowed when they come from verifiable data, e.g. "figures generated using code that has been reviewed and verified by the author". Disclose in an "AI Declaration in your manuscript" and also "within the caption or legend".

**Not found:**
- a fixed section for the declaration;
- a template;
- a specific rule on AI-written responses to reviewers.

Individual journals may add their own rules; I did not check any.

## 3. Elsevier

Source: elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals (no date shown).

**Disclosure:** "a separate AI declaration statement included in their manuscript upon submission", which will be published. Basic grammar checks are exempt. Research-process use goes "in detail in the Methods section".

**Code:** authors who "write or edit code using AI tools as part of their research … should declare this in detail in the Methods section".

**References:** AI may help organise literature, but use to "select, collate, generate or edit references … should be declared". Fabricated references "may lead to rejection".

**Figures:**
- Data visualisations must be "directly derived from underlying data", with the tool and version named in Methods.
- Explanatory images need a caption disclosure.
- Graphical abstracts may not be made with general-purpose image AI.

**Recommended format:**
- Section title: "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process".
- Statement: "During the preparation of this work, the author(s) used [NAME OF TOOL / SERVICE] in order to [REASON]. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the published article."

**Not found:** any author-side rule on reviewer reports or on responses to reviewers.

## 4. TMLR

Source: jmlr.org/tmlr/editorial-policies.html (no date shown).

- "LLMs may be used as general-purpose assistive tools … authors are fully responsible".
- Submissions or reviews that look "largely LLM-generated will be closely examined".
- "LLMs are not eligible for authorship."
- Reviewers may use LLMs for help such as "polishing the language" (reviewer guide).

**Not found:** any required disclosure, template, or rule on references or rebuttals.

## 5. Is the author's wording compliant?

| Publisher | As written | Smallest change needed |
|---|---|---|
| **IOP (MLST)** | **No**, for three reasons. (a) The statement does not explain how the outputs were verified or validated, which the policy requires; "reviewed and edited all content" does not cover checking the code and the references. (b) No tool version, which the template asks for. (c) The code use also has to appear in Methods. | Put it in the Acknowledgements as a separate AI statement. Add a version after "Claude", e.g. "(Claude, Anthropic; Claude Opus models)". Add one sentence that describes the checks as procedures, not as the author supervising the AI: "All code was checked by automated tests and by regenerating every reported number from the archived results files, and every cited source was checked by the author against the original." Add one sentence in Methods saying the analysis code was written with AI assistance and how it was tested. |
| **Elsevier** | **Yes in substance.** It is close to Elsevier's own template, plus more detail. | Use the recommended section title. Add the same one-sentence code statement to Methods, as the policy requires for AI-written code. |
| **Springer Nature** | **Yes in substance.** It describes the use and confirms the author's accountability. | Recommended: add the procedural verification sentence above, since SN asks for "validation steps" when AI touches analysis or code. Disclose in a figure caption only if a figure itself was made with AI. |
| **TMLR** | **Yes.** No disclosure is required, so including it is optional and harmless. | None. |

**Name the version from records.** Fill in the model version from the project's own records; I did not verify which Claude models were used.

**One change needed everywhere if a co-author is added.** The TÜBİTAK fee coverage at MLST needs a YTÜ faculty member as corresponding author. If one is added, "the author" has to become "the authors". The claims "designed the study, made every scientific decision" must also be true of the whole author group, or be given to the first author by name.

**Not a wording issue, but binding at IOP.** No AI-written reviewer responses and no reviewer reports uploaded to AI tools. SN also forbids putting reviewer reports into public AI systems.

The proposed changes keep the user's set text whole and only add facts, so the "change the journal, never the disclosure" rule is not triggered. The user still has to approve them.