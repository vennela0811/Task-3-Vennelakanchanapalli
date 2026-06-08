# ============================================================
# Tech Stack Recommender
# Capstone: Tech Stack Recommender
# Pipeline: User Skills → TF-IDF + Cosine Similarity → Top-N
# ============================================================

# --- IMPORTS ---
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PHASE 1 — INPUT: The Knowledge Base (Item Dataset)
# Job Roles as "items" — each has a skill tag document
# ============================================================

job_roles = {
    "Data Scientist": 
        "python machine-learning statistics pandas numpy tensorflow deep-learning data-analysis sql jupyter",

    "Machine Learning Engineer": 
        "python tensorflow pytorch machine-learning algorithms model-deployment docker kubernetes mlops",

    "Backend Developer": 
        "java python sql rest-api node-js databases docker git microservices system-design",

    "Frontend Developer": 
        "javascript react html css typescript ui-ux node-js figma responsive-design",

    "DevOps Engineer": 
        "aws docker kubernetes ci-cd linux shell-scripting git terraform ansible cloud-computing",

    "Data Engineer": 
        "python sql spark hadoop etl pipelines airflow kafka cloud-computing data-warehousing",

    "Cloud Architect": 
        "aws azure gcp cloud-computing docker kubernetes terraform networking security devops",

    "Cybersecurity Analyst": 
        "networking security linux penetration-testing firewalls encryption ethical-hacking siem soc",

    "Full Stack Developer": 
        "javascript react node-js python sql html css rest-api git docker databases",

    "AI Research Scientist": 
        "python pytorch tensorflow deep-learning nlp computer-vision research mathematics statistics algorithms",

    "Mobile Developer": 
        "flutter dart react-native android ios swift kotlin mobile-development api",

    "Database Administrator": 
        "sql mysql postgresql oracle databases indexing backup recovery performance-tuning data-modeling",
}

print("=" * 60)
print("   Tech Stack Recommender")
print("   The Digital Matchmaker Engine")
print("=" * 60)

print(f"\n[KNOWLEDGE BASE LOADED]")
print(f"  Total Job Roles : {len(job_roles)}")
print(f"  Algorithm       : TF-IDF Vectorization + Cosine Similarity")
print(f"  Pipeline        : Ingest → Score → Sort → Filter (Top-3)")

# ============================================================
# PHASE 1 — INPUT: Ingestion (User Cold Start Bypass)
# Minimum 3 skills required for sufficient data density
# ============================================================

print("\n" + "=" * 60)
print("   STEP 1: INGESTION — Capture User State")
print("=" * 60)
print("\n  Enter at least 3 skills (comma-separated).")
print("  Example: python, machine learning, sql, docker\n")

raw_input_text = input("  Your Skills: ")

# --- Sanitize: strip + lowercase (same as P1 logic) ---
user_skills_list = [s.strip().lower().replace(" ", "-")
                    for s in raw_input_text.split(",")
                    if s.strip()]

# --- Cold Start Guard: enforce minimum 3 inputs ---
if len(user_skills_list) < 3:
    print("\n  [WARNING] Less than 3 skills entered.")
    print("  Cold Start bypass: Adding generic fallback skills...")
    while len(user_skills_list) < 3:
        user_skills_list.append("software")

user_profile_string = " ".join(user_skills_list)

print(f"\n  [USER PROFILE BUILT]")
print(f"  Raw Skills     : {user_skills_list}")
print(f"  Profile Vector : '{user_profile_string}'")

# ============================================================
# PHASE 2 — PROCESS: TF-IDF Vectorization → Cosine Similarity
# Step 1: Ingestion ✓ (done above)
# Step 2: Scoring
# Step 3: Sorting
# Step 4: Filtering (Top-N)
# ============================================================

print("\n" + "=" * 60)
print("   STEP 2: SCORING — TF-IDF + Cosine Similarity")
print("=" * 60)

# --- Build the corpus: all job role documents + user profile ---
role_names     = list(job_roles.keys())
role_documents = list(job_roles.values())

# Append user profile to corpus for shared vocabulary space
corpus = role_documents + [user_profile_string]

# --- TF-IDF Vectorizer: penalizes generic, rewards specific ---
vectorizer  = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# Separate: item vectors vs user vector
item_vectors = tfidf_matrix[:-1]          # All job roles
user_vector  = tfidf_matrix[-1]           # User profile (last row)

print(f"\n  [TF-IDF APPLIED]")
print(f"  Vocabulary Size : {len(vectorizer.get_feature_names_out())} unique terms")
print(f"  Vector Shape    : {item_vectors.shape}")

# --- Cosine Similarity: angle-based, magnitude-invariant ---
scores = cosine_similarity(user_vector, item_vectors)[0]

print(f"\n  [COSINE SCORES COMPUTED]")
for role, score in zip(role_names, scores):
    bar = "█" * int(score * 30)
    print(f"  {role:<30} {score:.4f}  {bar}")

# ============================================================
# STEP 3: SORTING — Descending by score
# ============================================================

scored_roles = dict(zip(role_names, scores))
sorted_roles = dict(sorted(scored_roles.items(),
                           key=lambda x: x[1], reverse=True))

# ============================================================
# STEP 4: FILTERING — Top-N List (prevent choice overload)
# ============================================================

TOP_N = 3
top_roles = list(sorted_roles.items())[:TOP_N]

# ============================================================
# PHASE 3 — OUTPUT: Ranked Recommendations
# ============================================================

print("\n" + "=" * 60)
print("   STEP 3 & 4: OUTPUT — Top-N Recommendations")
print("=" * 60)

print(f"\n  Your Skills   : {', '.join(user_skills_list)}")
print(f"\n  TOP {TOP_N} CAREER PATH MATCHES:")
print("  " + "-" * 40)

medals = ["🥇", "🥈", "🥉"]
for i, (role, score) in enumerate(top_roles):
    pct  = score * 100
    bar  = "█" * int(pct / 3)
    print(f"\n  {medals[i]}  Rank {i+1}: {role}")
    print(f"      Match Score : {score:.4f} ({pct:.1f}%)")
    print(f"      Alignment   : [{bar:<33}]")
    print(f"      Skills Req  : {job_roles[role][:60]}...")

print("\n  " + "=" * 58)

# ============================================================
# VISUALIZATIONS
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle(
    f"DecodeLabs — Project 3: Tech Stack Recommender\n"
    f"User Skills: {', '.join(user_skills_list[:5])}",
    fontsize=13, fontweight='bold', y=1.02
)

# --- VIZ 1: Full Score Bar Chart ---
ax1    = axes[0]
roles  = list(sorted_roles.keys())
vals   = list(sorted_roles.values())
colors = ['#E8A838' if i < TOP_N else '#AAAAAA' for i in range(len(roles))]

bars = ax1.barh(roles[::-1], vals[::-1], color=colors[::-1],
                edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, vals[::-1]):
    ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
             f'{val:.3f}', va='center', fontsize=8, fontweight='bold')

ax1.set_title('All Roles — Cosine Similarity Scores\n(Orange = Top 3)',
              fontsize=11, fontweight='bold', pad=10)
ax1.set_xlabel('Cosine Similarity Score', fontsize=10)
ax1.axvline(x=top_roles[-1][1], color='#E8A838',
            linestyle='--', alpha=0.6, label='Top-3 cutoff')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3, axis='x')
ax1.set_xlim(0, max(vals) + 0.08)

# --- VIZ 2: Top-3 Podium Chart ---
ax2         = axes[1]
top_names   = [r[0].replace(" ", "\n") for r in top_roles]
top_scores  = [r[1] for r in top_roles]
podium_clrs = ['#E8A838', '#C0C0C0', '#CD7F32']

bars2 = ax2.bar(top_names, top_scores, color=podium_clrs,
                edgecolor='white', linewidth=1.5, width=0.5)
for bar, score in zip(bars2, top_scores):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.005,
             f'{score * 100:.1f}%',
             ha='center', fontsize=11, fontweight='bold')

ax2.set_title('Top-3 Career Matches\n(The Filtered Output)',
              fontsize=11, fontweight='bold', pad=10)
ax2.set_ylabel('Match Score', fontsize=10)
ax2.set_ylim(0, max(top_scores) + 0.1)
ax2.grid(True, alpha=0.3, axis='y')

for i, bar in enumerate(bars2):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             0.01, medals[i], ha='center', fontsize=18)

# --- VIZ 3: TF-IDF Heatmap (User vs Top-5 roles) ---
ax3 = axes[2]

top5_names = [r[0] for r in list(sorted_roles.items())[:5]]
top5_idx   = [role_names.index(n) for n in top5_names]

# Get top 10 user skill terms
feature_names = vectorizer.get_feature_names_out()
user_arr      = user_vector.toarray()[0]
top_term_idx  = np.argsort(user_arr)[::-1][:10]
top_terms     = feature_names[top_term_idx]

# Build sub-matrix: top-5 roles x top-10 user terms
sub_matrix = item_vectors[top5_idx].toarray()[:, top_term_idx]

sns.heatmap(sub_matrix,
            xticklabels=top_terms,
            yticklabels=[n.replace(" ", "\n") for n in top5_names],
            cmap='YlOrBr', annot=True, fmt='.2f',
            linewidths=0.5, linecolor='white',
            ax=ax3, annot_kws={"size": 8})

ax3.set_title('TF-IDF Weight Heatmap\n(Top-5 Roles × Your Key Skills)',
              fontsize=11, fontweight='bold', pad=10)
ax3.set_xlabel('Your Key Skill Terms', fontsize=9)
ax3.tick_params(axis='x', rotation=30, labelsize=8)
ax3.tick_params(axis='y', rotation=0, labelsize=8)

plt.tight_layout()
plt.savefig('viz_p3_recommender.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[VISUALIZATION SAVED] viz_p3_recommender.png")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("   PIPELINE SUMMARY")
print("=" * 60)
print(f"\n  Step 1 Ingestion  : {len(user_skills_list)} skills captured")
print(f"  Step 2 Scoring    : Cosine similarity vs {len(job_roles)} roles")
print(f"  Step 3 Sorting    : Descending rank order applied")
print(f"  Step 4 Filtering  : Top-{TOP_N} results delivered")
print(f"\n  Best Match        : {top_roles[0][0]} ({top_roles[0][1]*100:.1f}% aligned)")
print(f"\n[PIPELINE COMPLETE] DecodeLabs Project 3 — Done.")