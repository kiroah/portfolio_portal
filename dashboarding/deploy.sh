#!/bin/bash

# --- CONFIGURATION ---
REMOTE_USER="root"
REMOTE_HOST="37.27.29.49" 
SSH_KEY="~/.ssh/google_compute_engine" 

# Only the main domain is needed now
DOMAIN_MAIN="hironai.to"

echo "--- 🚀 Starting Deployment to $REMOTE_HOST ---"

# 1. Accept new server fingerprint
ssh-keygen -R $REMOTE_HOST > /dev/null 2>&1

# 2. Upload
echo "--- 📂 Uploading files... ---"
scp -i $SSH_KEY -o StrictHostKeyChecking=no -r ./apps ./configs ./data $REMOTE_USER@$REMOTE_HOST:/tmp/

# 3. Configure
echo "--- ⚙️  Configuring Server... ---"
ssh -i $SSH_KEY -o StrictHostKeyChecking=no $REMOTE_USER@$REMOTE_HOST << EOF

    export DEBIAN_FRONTEND=noninteractive
    apt-get update -q && apt-get install -yq nginx python3-venv python3-pip certbot python3-certbot-nginx

    # Setup Directory
    mkdir -p /var/www/apps
    mkdir -p /var/www/data
    cp -r /tmp/apps/* /var/www/apps/
    cp -r /tmp/data/* /var/www/data/
    
    # Virtual Env
    if [ ! -d "/var/www/apps/venv" ]; then
        python3 -m venv /var/www/apps/venv
    fi

    # Install Requirements
    /var/www/apps/venv/bin/pip install -r /var/www/apps/requirements.txt

    # Configs
    cp /tmp/configs/dash.service /etc/systemd/system/
    cp /tmp/configs/panel.service /etc/systemd/system/
    cp /tmp/configs/nginx.conf /etc/nginx/sites-available/default

    # Restart
    systemctl daemon-reload
    systemctl enable dash panel
    systemctl restart dash panel
    systemctl restart nginx
    
    rm -rf /tmp/apps /tmp/configs

EOF

echo "--- 🔒 Setting up SSL for $DOMAIN_MAIN... ---"

# 4. Certbot (Single Domain)
ssh -i $SSH_KEY $REMOTE_USER@$REMOTE_HOST "certbot --nginx --non-interactive --agree-tos -m admin@$DOMAIN_MAIN -d $DOMAIN_MAIN --redirect"

echo "--- 🎉 DONE! Visit https://$DOMAIN_MAIN/dash/ ---"