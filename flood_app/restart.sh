#!/bin/bash
# Restart Streamlit na porcie 8080. Uruchom: bash /home/ubuntu/flood_gif_app/restart.sh
pkill -f "[s]treamlit run"
sleep 2
cd /home/ubuntu/flood_gif_app || exit 1
setsid nohup ./venv/bin/streamlit run app.py --server.port 8080 > streamlit.log 2>&1 < /dev/null &
echo "Streamlit uruchomiony ponownie"
