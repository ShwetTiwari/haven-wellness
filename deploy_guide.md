# Haven — Deployment Guide (Render.com Free Tier)
# ================================================

## Why Render:
# - Free tier: 750 hours/month (enough for always-on)
# - Supports environment variables (API key stays secret)
# - Auto-deploys from GitHub
# - Free SSL certificate

## STEP 1 — Push to GitHub
# In your Depression_Detection folder:

git init
git add .
git commit -m "Haven wellness companion - initial deploy"

# Create a new repo on github.com (call it "haven-wellness")
# Then:
git remote add origin https://github.com/YOUR_USERNAME/haven-wellness.git
git push -u origin main

## STEP 2 — Create .gitignore (IMPORTANT - keeps API key safe)
# Already created below as .gitignore

## STEP 3 — Deploy on Render
# 1. Go to https://render.com and sign up (free)
# 2. Click "New +" → "Web Service"
# 3. Connect your GitHub repo
# 4. Settings:
#    Name: haven-wellness
#    Runtime: Python 3
#    Build Command: pip install -r requirements.txt
#    Start Command: gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
# 5. Add Environment Variables:
#    ANTHROPIC_API_KEY = your_key_here
#    SECRET_KEY = any_random_string_here
#    DEBUG = false
# 6. Click "Create Web Service"
# 7. Wait ~5 minutes for build
# 8. Your app is live at: https://haven-wellness.onrender.com

## IMPORTANT NOTES FOR FREE TIER:
# - Free tier sleeps after 15 min inactivity (first request takes ~30s to wake)
# - Models (.pkl files) must be committed to git (they're needed for inference)
# - Audio/video processing works fine
# - Add a "Keep alive" ping if you want it always awake

## STEP 4 — Keep models in git
# The .pkl files in models/saved/ need to be in git
# They're binary but small enough (~5MB each)
# Make sure .gitignore does NOT exclude them