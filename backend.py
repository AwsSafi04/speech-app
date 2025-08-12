# Load Whisper AI model
print("Loading Whisper model...")

try:
    # Determine device and settings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    print(f"Using device: {device}")
    print(f"Compute type: {compute_type}")
    
    # Load the model
    model = WhisperModel("large-v3", device=device, compute_type=compute_type)
    
    print("Whisper large-v3 model loaded successfully!")
    print("Ready for Arabic, English, and 90+ languages!")
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying with smaller model...")
    try:
        model = WhisperModel("base", device="cpu", compute_type="int8")
        print("Backup model loaded (may be less accurate)")
    except:
        model = None
        print("Model loading failed completely")
