# vision-llm-safety-system

Real-time safety monitoring pipeline that combines computer vision
detection with LLM-generated incident reporting — detections become
structured events, events become natural-language safety summaries.

> 🚧 In active development — building in public, daily commits.

## Roadmap
- [x] Webcam person detection with event snapshots (YOLOv5)
- [ ] Alert deduplication (cooldown between events)
- [ ] SQLite event logging + query CLI
- [ ] PPE violation detection on video
- [ ] Rules engine: zone + violation → risk level
- [ ] LLM incident summaries & shift-handover reports (structured JSON)
- [ ] Ops chatbot over incident history (RAG)
- [ ] Streamlit dashboard + demo video

## Quick start

    # create virtual environment
    python -m venv venv
    [LINUX] source venv/bin/activate
    [OR]
    [WINDOWS] .\venv\Scripts\activate
    
    # install dependencies
    pip install opencv-python torch torchvision ultralytics pandas tqdm seaborn scipy
    [OR]
    curl -O https://raw.githubusercontent.com/ultralytics/yolov5/refs/heads/master/requirements.txt
    pip install -r requirements.txt
    
    python detector.py
    
Press `q` to quit. Detected events are saved as timestamped snapshots.

## Design notes
- Model loads once at startup; camera stream processed frame-by-frame
- Detection thresholds are named constants (soon: config file)
- Safety logic will stay deterministic — the LLM narrates, it never decides
