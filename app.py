import os
import logging
from typing import Dict, Any, Tuple
from flask import Flask, request, jsonify, render_template, Response, Request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError

# Import your custom modules
try:
    from fact_verification import verify_fact, detect_deepfake, evaluate_research_paper
except ImportError:
    logging.error("Failed to import from fact_verification.py. Make sure the file exists and has the required functions.")
    exit("Exiting due to missing fact_verification module.")

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(name)s:%(lineno)d: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# --- Configuration ---
UPLOAD_FOLDER: str = 'uploads'
MAX_FILE_SIZE_MB: int = 20
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_MB * 1024 * 1024

# Ensure the upload folder exists
try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.logger.info(f"Upload folder '{UPLOAD_FOLDER}' ensured.")
except OSError as e:
    app.logger.critical(f"Could not create upload folder '{UPLOAD_FOLDER}': {e}", exc_info=True)
    exit(f"Failed to create upload folder. Exiting.")

# --- Allowed File Extensions ---
ALLOWED_IMAGE_EXTENSIONS: set = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
ALLOWED_PDF_EXTENSIONS: set = {'.pdf'}

# --- Helper Function ---
def is_allowed_file(filename: str, allowed_extensions: set) -> bool:
    return '.' in filename and os.path.splitext(filename)[1].lower() in allowed_extensions

# --- Routes ---

@app.route("/")
def home() -> str | Tuple[str, int]:
    try:
        return render_template("index.html")
    except Exception as e:
        app.logger.error(f"Error rendering index.html: {e}", exc_info=True)
        return "Error loading page. Please check server logs.", 500

@app.route("/verify", methods=["POST"])
def fact_check_route() -> Response | Tuple[Response, int]:
    req: Request = request
    try:
        data: Dict[str, Any] | None = req.get_json()
        if not data or "claim" not in data:
            app.logger.warning("Fact check request missing JSON or 'claim' field.")
            return jsonify({"error": "Request must be JSON with a 'claim' field."}), 400

        claim: str = data.get("claim", "").strip()
        if not claim:
            app.logger.warning("Fact check request received with empty claim.")
            return jsonify({"error": "No claim provided"}), 400

        app.logger.info(f"Received fact check request for claim: '{claim[:100]}...'")

        truth_score, explanation, sources = verify_fact(claim)
        app.logger.info(f"Fact check result - Score: {truth_score}")

        return jsonify({
            "type": "fact_check",
            "truth_score": truth_score,
            "explanation": explanation,
            "sources": sources or []
        })
    except Exception as e:
        app.logger.error(f"Error during fact verification process", exc_info=True)
        return jsonify({"error": "An internal server error occurred during fact-checking."}), 500

@app.route("/detect", methods=["POST"])
def image_analysis_route() -> Response | Tuple[Response, int]:
    req: Request = request
    if "image" not in req.files:
        app.logger.warning("Image detection request missing 'image' file part.")
        return jsonify({"error": "No image file part in the request"}), 400

    file: FileStorage | None = req.files.get("image")
    if not file or not file.filename:
        app.logger.warning("Image detection request received with no file selected or empty filename.")
        return jsonify({"error": "No file selected for upload"}), 400

    filename: str = secure_filename(file.filename)
    file_path: str | None = None

    if not is_allowed_file(filename, ALLOWED_IMAGE_EXTENSIONS):
        ext: str = os.path.splitext(filename)[1]
        app.logger.warning(f"Image detection attempt with invalid file type: {filename}")
        return jsonify({"error": f"Invalid file type ({ext}). Please upload an image ({', '.join(sorted(list(ALLOWED_IMAGE_EXTENSIONS)))})."}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    app.logger.info(f"Processing image upload: {filename}")

    try:
        file.save(file_path)
        app.logger.info(f"Saved temporary image file: {file_path}")
        app.logger.info(f"Starting image authenticity analysis for: {filename}")
        result: Dict[str, Any] = detect_deepfake(file_path)
        app.logger.info(f"Image analysis complete for: {filename}")
        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Error processing image {filename}", exc_info=True)
        return jsonify({"type": "deepfake_detection", "error": "An internal server error occurred while processing the image."}), 500
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                app.logger.info(f"Removed temporary image file: {file_path}")
            except OSError as remove_error:
                app.logger.warning(f"Error removing temporary image file {file_path}: {remove_error}", exc_info=True)

@app.route("/detect_pdf", methods=["POST"])
def detect_pdf_route() -> Response | Tuple[Response, int]:
    req: Request = request
    if "pdf" not in req.files:
        app.logger.warning("PDF detection request missing 'pdf' file part.")
        return jsonify({"error": "No PDF file part in the request"}), 400

    file: FileStorage | None = req.files.get("pdf")
    if not file or not file.filename:
        app.logger.warning("PDF detection request received with no file selected or empty filename.")
        return jsonify({"error": "No file selected for upload"}), 400

    filename: str = secure_filename(file.filename)
    file_path: str | None = None

    if not is_allowed_file(filename, ALLOWED_PDF_EXTENSIONS):
        app.logger.warning(f"PDF detection attempt with non-PDF file: {filename}")
        return jsonify({"error": "Invalid file type. Please upload a PDF file."}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    app.logger.info(f"Processing PDF upload: {filename}")

    extracted_text: str | None = None
    preview_text: str = ""

    try:
        file.save(file_path)
        app.logger.info(f"Saved temporary PDF file: {file_path}")

        try:
            app.logger.info(f"Attempting text extraction from PDF: {filename}...")
            extracted_text = extract_text(file_path)
            if extracted_text:
                preview_text = extracted_text[:500] + ('...' if len(extracted_text) > 500 else '')
                app.logger.info(f"Text extraction successful from {filename}, {len(extracted_text)} characters extracted.")
            else:
                app.logger.info(f"Text extraction from {filename} resulted in empty string.")
                preview_text = ""
        except PDFSyntaxError as pdf_err:
            app.logger.error(f"Syntax error extracting text from PDF {filename}: {pdf_err}", exc_info=True)
            return jsonify({"error": f"Could not extract text. PDF file seems corrupted or invalid: {pdf_err}"}), 400
        except Exception as extraction_error:
            app.logger.error(f"Error extracting text from PDF {filename}", exc_info=True)
            return jsonify({"error": f"Could not extract text from PDF (check if image-based or encrypted)."}), 500

        if not extracted_text or not extracted_text.strip():
            app.logger.info(f"No text extracted or text is whitespace for {filename}. Skipping evaluation.")
            return jsonify({
                "type": "evaluation",
                "preview": preview_text,
                "score_percent": "N/A",
                "justification": "No text could be extracted from the PDF."
            }), 200

        app.logger.info(f"Calling evaluate_research_paper for PDF: {filename}...")
        score_percent, justification = evaluate_research_paper(extracted_text)
        app.logger.info(f"Evaluation complete for {filename}. Score: {score_percent}")

        return jsonify({
            "type": "evaluation",
            "preview": preview_text,
            "score_percent": score_percent,
            "justification": justification
        })

    except Exception as e:
        app.logger.error(f"Unexpected error processing PDF {filename}", exc_info=True)
        return jsonify({"error": "An unexpected internal error occurred while processing the PDF."}), 500
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                app.logger.info(f"Removed temporary PDF file: {file_path}")
            except OSError as remove_error:
                app.logger.warning(f"Error removing temporary PDF file {file_path}: {remove_error}", exc_info=True)

# --- Main Execution ---
if __name__ == "__main__":
    app.logger.info("Starting Flask development server...")
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
