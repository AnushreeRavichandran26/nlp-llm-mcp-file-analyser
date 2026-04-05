#!/usr/bin/env python3

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import base64
import mimetypes
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniversalMCPFileServer:
    def __init__(self, allowed_directory: str = None, restrict_to_directory: bool = False):
        """
        Initialize Universal MCP File Server with support for any file type
        
        Args:
            allowed_directory: Base directory for operations (default: None for full system access)
            restrict_to_directory: If True, restrict operations to allowed_directory only
        """
        self.restrict_to_directory = restrict_to_directory
        self.document_context = {}  # Store document contents for context
        self.current_user = "anucodez"
        self.current_time = "2025-06-08 20:38:53"
        
        if allowed_directory:
            self.allowed_directory = Path(allowed_directory).resolve()
        else:
            # Full system access - use root directory
            if os.name == 'nt':  # Windows
                self.allowed_directory = Path('C:/')
            else:  # Unix-like systems
                self.allowed_directory = Path('/')
        
        self.server_info = {
            "name": "universal-file-server",
            "version": "1.0.0"
        }
        self.capabilities = {
            "tools": {}
        }
        
        logger.info(f"Universal MCP File Server initialized")
        logger.info(f"User: {self.current_user}")
        logger.info(f"Time: {self.current_time} UTC")
        logger.info(f"Base directory: {self.allowed_directory}")
        logger.info(f"Restricted mode: {self.restrict_to_directory}")
        
    def resolve_path(self, file_path: str) -> Path:
        """Resolve file path to absolute path"""
        if os.path.isabs(file_path):
            return Path(file_path).resolve()
        else:
            return (self.allowed_directory / file_path).resolve()
    
    def get_current_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    def extract_text_from_any_file(self, file_path: Path) -> Dict[str, Any]:
        """Extract text content from any file type"""
        file_extension = file_path.suffix.lower()
        file_size = file_path.stat().st_size
        
        result = {
            "success": False,
            "content": "",
            "file_type": file_extension,
            "original_size": file_size,
            "extracted_size": 0,
            "method": "unknown",
            "error": None
        }
        
        try:
            # PDF files
            if file_extension == '.pdf':
                result.update(self._extract_pdf(file_path))
            
            # Microsoft Office files
            elif file_extension == '.docx':
                result.update(self._extract_docx(file_path))
            elif file_extension == '.xlsx':
                result.update(self._extract_xlsx(file_path))
            elif file_extension == '.pptx':
                result.update(self._extract_pptx(file_path))
            
            # Text-based files
            elif file_extension in ['.txt', '.md', '.markdown', '.rst', '.log']:
                result.update(self._extract_text_file(file_path))
            
            # Code files
            elif file_extension in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go', '.rs', '.swift']:
                result.update(self._extract_code_file(file_path))
            
            # Data files
            elif file_extension in ['.json', '.yaml', '.yml', '.xml', '.csv', '.tsv']:
                result.update(self._extract_data_file(file_path))
            
            # Config files
            elif file_extension in ['.ini', '.conf', '.config', '.cfg', '.toml']:
                result.update(self._extract_config_file(file_path))
            
            # Web files
            elif file_extension in ['.html', '.htm', '.css', '.scss', '.sass']:
                result.update(self._extract_web_file(file_path))
            
            # Script files
            elif file_extension in ['.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd']:
                result.update(self._extract_script_file(file_path))
            
            # Legacy Office files
            elif file_extension in ['.doc', '.xls', '.ppt']:
                result.update(self._extract_legacy_office(file_path))
            
            # Image files (extract metadata)
            elif file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
                result.update(self._extract_image_metadata(file_path))
            
            # Binary files (show info only)
            else:
                result.update(self._extract_binary_info(file_path))
                
        except Exception as e:
            result["error"] = str(e)
            result["content"] = f"Error extracting content: {str(e)}"
        
        result["extracted_size"] = len(result["content"])
        return result
    
    def _extract_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from PDF"""
        try:
            import PyPDF2
            
            content = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        content += f"\n--- Page {page_num + 1} (Error: {str(e)}) ---\n"
            
            return {
                "success": True,
                "content": content.strip(),
                "method": "PyPDF2",
                "pages": len(pdf_reader.pages)
            }
            
        except ImportError:
            return {
                "success": False,
                "content": "PDF support not available. Install with: pip install PyPDF2",
                "method": "missing_dependency",
                "error": "PyPDF2 not installed"
            }
        except Exception as e:
            return {
                "success": False,
                "content": f"Error reading PDF: {str(e)}",
                "method": "PyPDF2",
                "error": str(e)
            }
    
    def _extract_docx(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from DOCX"""
        try:
            import docx
            
            doc = docx.Document(file_path)
            content = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    content.append(paragraph.text)
            
            # Extract table content
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        content.append(" | ".join(row_text))
            
            return {
                "success": True,
                "content": "\n".join(content),
                "method": "python-docx",
                "paragraphs": len(doc.paragraphs),
                "tables": len(doc.tables)
            }
            
        except ImportError:
            return {
                "success": False,
                "content": "DOCX support not available. Install with: pip install python-docx",
                "method": "missing_dependency",
                "error": "python-docx not installed"
            }
        except Exception as e:
            return {
                "success": False,
                "content": f"Error reading DOCX: {str(e)}",
                "method": "python-docx",
                "error": str(e)
            }
    
    def _extract_xlsx(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from Excel files"""
        try:
            import openpyxl
            
            workbook = openpyxl.load_workbook(file_path, data_only=True)
            content = []
            
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                content.append(f"\n=== Sheet: {sheet_name} ===")
                
                for row in sheet.iter_rows(values_only=True):
                    row_text = []
                    for cell in row:
                        if cell is not None:
                            row_text.append(str(cell))
                    if any(cell.strip() for cell in row_text if isinstance(cell, str)):
                        content.append(" | ".join(row_text))
            
            return {
                "success": True,
                "content": "\n".join(content),
                "method": "openpyxl",
                "sheets": len(workbook.sheetnames)
            }
            
        except ImportError:
            return {
                "success": False,
                "content": "Excel support not available. Install with: pip install openpyxl",
                "method": "missing_dependency",
                "error": "openpyxl not installed"
            }
        except Exception as e:
            return {
                "success": False,
                "content": f"Error reading Excel file: {str(e)}",
                "method": "openpyxl",
                "error": str(e)
            }
    
    def _extract_pptx(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from PowerPoint files"""
        try:
            import pptx
            
            presentation = pptx.Presentation(file_path)
            content = []
            
            for slide_num, slide in enumerate(presentation.slides, 1):
                content.append(f"\n=== Slide {slide_num} ===")
                
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        content.append(shape.text)
            
            return {
                "success": True,
                "content": "\n".join(content),
                "method": "python-pptx",
                "slides": len(presentation.slides)
            }
            
        except ImportError:
            return {
                "success": False,
                "content": "PowerPoint support not available. Install with: pip install python-pptx",
                "method": "missing_dependency",
                "error": "python-pptx not installed"
            }
        except Exception as e:
            return {
                "success": False,
                "content": f"Error reading PowerPoint file: {str(e)}",
                "method": "python-pptx",
                "error": str(e)
            }
    
    def _extract_text_file(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from text files"""
        try:
            encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    content = file_path.read_text(encoding=encoding)
                    return {
                        "success": True,
                        "content": content,
                        "method": f"text_file_{encoding}",
                        "encoding": encoding
                    }
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, read as binary and show hex
            binary_content = file_path.read_bytes()
            return {
                "success": False,
                "content": f"Binary content (first 1000 bytes): {binary_content[:1000].hex()}",
                "method": "binary_fallback",
                "error": "Could not decode as text"
            }
            
        except Exception as e:
            return {
                "success": False,
                "content": f"Error reading text file: {str(e)}",
                "method": "text_file",
                "error": str(e)
            }
    
    def _extract_code_file(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from code files"""
        result = self._extract_text_file(file_path)
        if result["success"]:
            result["method"] = f"code_file_{file_path.suffix}"
            # Add syntax highlighting info
            result["language"] = file_path.suffix[1:]  # Remove the dot
        return result
    
    def _extract_data_file(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from data files (JSON, CSV, etc.)"""
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.json':
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                content = json.dumps(data, indent=2, ensure_ascii=False)
                return {
                    "success": True,
                    "content": content,
                    "method": "json_parser"
                }
            
            elif file_extension in ['.csv', '.tsv']:
                content = file_path.read_text(encoding='utf-8')
                return {
                    "success": True,
                    "content": content,
                    "method": f"{file_extension[1:]}_file"
                }
            
            elif file_extension in ['.yaml', '.yml']:
                try:
                    import yaml
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                    content = yaml.dump(data, default_flow_style=False, allow_unicode=True)
                    return {
                        "success": True,
                        "content": content,
                        "method": "yaml_parser"
                    }
                except ImportError:
                    return self._extract_text_file(file_path)
            
            elif file_extension == '.xml':
                content = file_path.read_text(encoding='utf-8')
                return {
                    "success": True,
                    "content": content,
                    "method": "xml_file"
                }
            
            else:
                return self._extract_text_file(file_path)
                
        except Exception as e:
            return {
                "success": False,
                "content": f"Error reading data file: {str(e)}",
                "method": f"data_file_{file_extension}",
                "error": str(e)
            }
    
    def _extract_config_file(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from configuration files"""
        result = self._extract_text_file(file_path)
        if result["success"]:
            result["method"] = f"config_file_{file_path.suffix}"
        return result
    
    def _extract_web_file(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from web files"""
        result = self._extract_text_file(file_path)
        if result["success"]:
            result["method"] = f"web_file_{file_path.suffix}"
        return result
    
    def _extract_script_file(self, file_path: Path) -> Dict[str, Any]:
        """Extract content from script files"""
        result = self._extract_text_file(file_path)
        if result["success"]:
            result["method"] = f"script_file_{file_path.suffix}"
        return result
    
    def _extract_legacy_office(self, file_path: Path) -> Dict[str, Any]:
        """Handle legacy Office files"""
        return {
            "success": False,
            "content": f"Legacy Office file ({file_path.suffix}). Please convert to modern format (.docx, .xlsx, .pptx) or PDF for text extraction.",
            "method": "legacy_office",
            "error": "Legacy format not supported"
        }
    
    def _extract_image_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from image files"""
        try:
            stat = file_path.stat()
            content = f"""Image File: {file_path.name}
Format: {file_path.suffix.upper()}
Size: {stat.st_size} bytes
Modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}

Note: This is an image file. Text extraction not applicable.
For image analysis, consider using OCR tools or image processing libraries."""
            
            return {
                "success": True,
                "content": content,
                "method": "image_metadata"
            }
        except Exception as e:
            return {
                "success": False,
                "content": f"Error reading image metadata: {str(e)}",
                "method": "image_metadata",
                "error": str(e)
            }
    
    def _extract_binary_info(self, file_path: Path) -> Dict[str, Any]:
        """Handle binary files"""
        try:
            stat = file_path.stat()
            
            # Try to determine file type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            # Read first few bytes for analysis
            with open(file_path, 'rb') as f:
                header = f.read(256)
            
            content = f"""Binary File: {file_path.name}
MIME Type: {mime_type or 'unknown'}
Size: {stat.st_size} bytes
Modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}

File Header (first 256 bytes):
{header.hex()[:512]}...

Note: This is a binary file. Direct text extraction not possible.
Consider converting to a text-readable format if needed."""
            
            return {
                "success": True,
                "content": content,
                "method": "binary_analysis",
                "mime_type": mime_type
            }
        except Exception as e:
            return {
                "success": False,
                "content": f"Error analyzing binary file: {str(e)}",
                "method": "binary_analysis",
                "error": str(e)
            }

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP requests"""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": self.capabilities,
                        "serverInfo": self.server_info
                    }
                }
            
            elif method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": "load_any_document",
                                "description": "Load and extract content from ANY file type (PDF, DOCX, TXT, images, code, data files, etc.) for context-aware questioning",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "path": {
                                            "type": "string",
                                            "description": "Full path to any file (PDF, DOCX, TXT, JPG, JSON, Python, etc.)"
                                        },
                                        "context_name": {
                                            "type": "string",
                                            "description": "Optional name to identify this document in context (defaults to filename)"
                                        }
                                    },
                                    "required": ["path"]
                                }
                            },
                            {
                                "name": "query_document_context",
                                "description": "Query the loaded document context to answer questions about any loaded files",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "query": {
                                            "type": "string",
                                            "description": "Question or query about the loaded documents"
                                        },
                                        "document_name": {
                                            "type": "string",
                                            "description": "Optional: specific document to query (if multiple loaded)"
                                        }
                                    },
                                    "required": ["query"]
                                }
                            },
                            {
                                "name": "list_loaded_documents",
                                "description": "List all currently loaded documents in context",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {},
                                    "required": []
                                }
                            },
                            {
                                "name": "clear_document_context",
                                "description": "Clear all loaded documents from context",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "document_name": {
                                            "type": "string",
                                            "description": "Optional: specific document to clear (if not provided, clears all)"
                                        }
                                    },
                                    "required": []
                                }
                            },
                            {
                                "name": "read_file",
                                "description": "Read and analyze ANY file type with intelligent content extraction",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "path": {
                                            "type": "string",
                                            "description": "Full path to any file type"
                                        }
                                    },
                                    "required": ["path"]
                                }
                            },
                            {
                                "name": "write_file",
                                "description": "Write content to a file (supports full system paths)",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "path": {
                                            "type": "string",
                                            "description": "Full path to the file to write"
                                        },
                                        "content": {
                                            "type": "string",
                                            "description": "Content to write to the file"
                                        }
                                    },
                                    "required": ["path", "content"]
                                }
                            },
                            {
                                "name": "list_directory",
                                "description": "List contents of a directory with file type detection",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "path": {
                                            "type": "string",
                                            "description": "Directory path to list"
                                        }
                                    },
                                    "required": ["path"]
                                }
                            },
                            {
                                "name": "find_files",
                                "description": "Find files of any type matching patterns",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "directory": {
                                            "type": "string",
                                            "description": "Directory to search in"
                                        },
                                        "pattern": {
                                            "type": "string",
                                            "description": "File pattern (*.pdf, *.txt, *.py, etc.)"
                                        },
                                        "recursive": {
                                            "type": "boolean",
                                            "description": "Search subdirectories",
                                            "default": True
                                        }
                                    },
                                    "required": ["directory", "pattern"]
                                }
                            }
                        ]
                    }
                }
            
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                if tool_name == "load_any_document":
                    return await self.load_any_document(request_id, arguments)
                elif tool_name == "query_document_context":
                    return await self.query_document_context(request_id, arguments)
                elif tool_name == "list_loaded_documents":
                    return await self.list_loaded_documents(request_id, arguments)
                elif tool_name == "clear_document_context":
                    return await self.clear_document_context(request_id, arguments)
                elif tool_name == "read_file":
                    return await self.read_any_file(request_id, arguments)
                elif tool_name == "write_file":
                    return await self.write_file(request_id, arguments)
                elif tool_name == "list_directory":
                    return await self.list_directory(request_id, arguments)
                elif tool_name == "find_files":
                    return await self.find_files(request_id, arguments)
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Unknown tool: {tool_name}"
                        }
                    }
            
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown method: {method}"
                    }
                }
                
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }
    
    async def load_any_document(self, request_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Load any file type and extract content for context"""
        file_path = arguments.get("path")
        context_name = arguments.get("context_name")
        
        if not file_path:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "File path is required"
                }
            }
        
        try:
            full_path = self.resolve_path(file_path)
            
            if not full_path.exists():
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": f"File not found: {full_path}"
                    }
                }
            
            if not full_path.is_file():
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": f"Path is not a file: {full_path}"
                    }
                }
            
            # Extract content using universal extractor
            extraction_result = self.extract_text_from_any_file(full_path)
            
            # Use filename as context name if not provided
            if not context_name:
                context_name = full_path.name
            
            # Store in document context
            self.document_context[context_name] = {
                "path": str(full_path),
                "content": extraction_result["content"],
                "file_type": extraction_result["file_type"],
                "extraction_method": extraction_result["method"],
                "extraction_success": extraction_result["success"],
                "original_size": extraction_result["original_size"],
                "extracted_size": extraction_result["extracted_size"],
                "loaded_at": self.get_current_timestamp(),
                "loaded_by": self.current_user
            }
            
            # Create response
            status = "✅ Successfully" if extraction_result["success"] else "⚠️ Partially"
            method_info = f"Method: {extraction_result['method']}"
            
            if "error" in extraction_result and extraction_result["error"]:
                method_info += f" (Note: {extraction_result['error']})"
            
            content_preview = extraction_result["content"][:500] + "..." if len(extraction_result["content"]) > 500 else extraction_result["content"]
            
            response_text = f"""✅ Successfully loaded document!

📄 Document: {context_name}
📁 File: {full_path}
📋 Type: {extraction_result['file_type'].upper()}
📏 Original Size: {extraction_result['original_size']} bytes
📝 Extracted Size: {extraction_result['extracted_size']} characters
🔧 Method: {extraction_result['method']}

📖 Content Preview:
{content_preview}

🎯 You can now ask questions about this document using natural language!"""

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": response_text
                        }
                    ]
                }
            }
            
        except PermissionError:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Permission denied: {file_path}"
                }
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Error loading document: {str(e)}"
                }
            }
    
    async def read_any_file(self, request_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Read any file type with intelligent content extraction"""
        file_path = arguments.get("path")
        
        if not file_path:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "File path is required"
                }
            }
        
        try:
            full_path = self.resolve_path(file_path)
            
            if not full_path.exists():
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": f"File not found: {full_path}"
                    }
                }
            
            if not full_path.is_file():
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": f"Path is not a file: {full_path}"
                    }
                }
            
            # Extract content
            extraction_result = self.extract_text_from_any_file(full_path)
            
            status = "✅ Successfully read" if extraction_result["success"] else "⚠️ Partially read"
            method_info = f"Method: {extraction_result['method']}"
            
            if "error" in extraction_result and extraction_result["error"]:
                method_info += f" (Note: {extraction_result['error']})"
            
            response_text = f"""{status} file!

📄 File: {full_path}
📋 Type: {extraction_result['file_type'].upper()}
📏 Size: {extraction_result['original_size']} bytes → {extraction_result['extracted_size']} characters
🔧 {method_info}
⏰ Read: {self.get_current_timestamp()} UTC

📖 Content:
{extraction_result['content']}"""

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": response_text
                        }
                    ]
                }
            }
            
        except PermissionError:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Permission denied: {file_path}"
                }
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Error reading file: {str(e)}"
                }
            }

    async def query_document_context(self, request_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Query the loaded document context"""
        query = arguments.get("query")
        document_name = arguments.get("document_name")
        
        if not query:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "Query is required"
                }
            }
        
        if not self.document_context:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": "No documents are currently loaded. Please use the 'load_any_document' tool to load a document first."
                        }
                    ]
                }
            }
        
        try:
            # Prepare context for the query
            context_info = []
            relevant_content = []
            
            if document_name and document_name in self.document_context:
                # Query specific document
                doc = self.document_context[document_name]
                context_info.append(f"📄 Document: {document_name} ({doc['file_type']}) - {doc['extraction_method']}")
                relevant_content.append(f"Content from {document_name}:\n{doc['content']}")
            else:
                # Query all loaded documents
                for name, doc in self.document_context.items():
                    context_info.append(f"📄 Document: {name} ({doc['file_type']}) - {doc['extraction_method']}")
                    relevant_content.append(f"Content from {name}:\n{doc['content']}")
            
            context_text = "\n".join(context_info)
            full_content = "\n\n" + "="*50 + "\n\n".join(relevant_content)
            
            response_text = f"""🔍 Query: "{query}"

📚 Available Documents:
{context_text}

⏰ Query Time: {self.get_current_timestamp()} UTC

📖 Document Content for Analysis:
{full_content}

💡 Note: This is the raw document content. The AI assistant will analyze this information to answer your specific question."""

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": response_text
                        }
                    ]
                }
            }
            
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Error querying document context: {str(e)}"
                }
            }
    
    async def list_loaded_documents(self, request_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """List all loaded documents"""
        if not self.document_context:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"📚 No documents currently loaded.\n\n⏰ Current Time: {self.get_current_timestamp()} UTC"
                        }
                    ]
                }
            }
        
        doc_list = []
        total_size = 0
        
        for name, doc in self.document_context.items():
            doc_list.append(f"📄 {name}")
            doc_list.append(f"   📁 Path: {doc['path']}")
            doc_list.append(f"   📋 Type: {doc['file_type'].upper()}")
            doc_list.append(f"   📏 Size: {doc['extracted_size']} characters")
            doc_list.append(f"   🔧 Method: {doc['extraction_method']}")
            doc_list.append(f"   ✅ Success: {'Yes' if doc['extraction_success'] else 'Partial'}")
            doc_list.append(f"   ⏰ Loaded: {doc['loaded_at']} UTC")
            #doc_list.append(f"   👤 User: {doc['loaded_by']}")
            doc_list.append("")
            total_size += doc['extracted_size']
        
        response_text = f"""📚 Loaded Documents ({len(self.document_context)}):

{chr(10).join(doc_list)}
📊 Total Content: {total_size} characters
⏰ Current Time: {self.get_current_timestamp()} UTC"""

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": response_text
                    }
                ]
            }
        }
    
    async def clear_document_context(self, request_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Clear document context"""
        document_name = arguments.get("document_name")
        
        if document_name:
            if document_name in self.document_context:
                del self.document_context[document_name]
                message = f"🗑️ Cleared document: {document_name}"
            else:
                message = f"❌ Document not found: {document_name}"
        else:
            count = len(self.document_context)
            self.document_context.clear()
            message = f"🗑️ Cleared all {count} loaded documents"
        
        message += f"\n\n⏰ Time: {self.get_current_timestamp()} UTC"
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": message
                    }
                ]
            }
        }

    # Include other methods (write_file, list_directory, find_files) from previous version...
    async def write_file(self, request_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Write content to file"""
        file_path = arguments.get("path")
        content = arguments.get("content")
        
        if not file_path:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "File path is required"
                }
            }
        
        if content is None:
            content = ""
        
        try:
            full_path = self.resolve_path(file_path)
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"✅ Successfully wrote {len(content)} characters to {full_path}\n\n⏰ Time: {self.get_current_timestamp()} UTC"
                        }
                    ]
                }
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Error writing file: {str(e)}"
                }
            }

    async def list_directory(self, request_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """List directory contents with file type detection"""
        dir_path = arguments.get("path", ".")
        
        try:
            full_path = self.resolve_path(dir_path)
            
            if not full_path.exists():
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": f"Directory not found: {full_path}"
                    }
                }
            
            if not full_path.is_dir():
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": f"Path is not a directory: {full_path}"
                    }
                }
            
            items = []
            file_types = {}
            
            for item in sorted(full_path.iterdir()):
                try:
                    stat = item.stat()
                    size = stat.st_size
                    item_type = "📁 DIR" if item.is_dir() else "📄 FILE"
                    
                    # Count file types
                    if item.is_file():
                        ext = item.suffix.lower()
                        file_types[ext] = file_types.get(ext, 0) + 1
                    
                    # Format size
                    if item.is_dir():
                        size_str = "<DIR>"
                    elif size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024*1024:
                        size_str = f"{size/1024:.1f}KB"
                    else:
                        size_str = f"{size/(1024*1024):.1f}MB"
                    
                    # Add file type emoji
                    ext = item.suffix.lower()
                    if item.is_dir():
                        emoji = "📁"
                    elif ext in ['.pdf']:
                        emoji = "📕"
                    elif ext in ['.docx', '.doc']:
                        emoji = "📘"
                    elif ext in ['.xlsx', '.xls']:
                        emoji = "📗"
                    elif ext in ['.pptx', '.ppt']:
                        emoji = "📙"
                    elif ext in ['.txt', '.md']:
                        emoji = "📄"
                    elif ext in ['.py', '.js', '.java']:
                        emoji = "💻"
                    elif ext in ['.jpg', '.png', '.gif']:
                        emoji = "🖼️"
                    else:
                        emoji = "📄"
                    
                    items.append(f"{emoji} {item.name:<40} {item_type:<10} {size_str}")
                except Exception:
                    items.append(f"❓ {item.name:<40} {'unknown':<10} {'?'}")
            
            # Create file type summary
            type_summary = []
            for ext, count in sorted(file_types.items()):
                if ext:
                    type_summary.append(f"{ext}: {count}")
                else:
                    type_summary.append(f"no extension: {count}")
            
            response_text = f"""📁 Directory: {full_path}
📊 Total items: {len(items)}
📋 File types: {', '.join(type_summary) if type_summary else 'No files'}
⏰ Listed: {self.get_current_timestamp()} UTC


{'Icon'} {'Name':<40} {'Type':<10} {'Size'}
{'-'*65}
{chr(10).join(items)}"""

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": response_text
                        }
                    ]
                }
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Error listing directory: {str(e)}"
                }
            }

    async def find_files(self, request_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Find files of any type matching patterns"""
        directory = arguments.get("directory")
        pattern = arguments.get("pattern")
        recursive = arguments.get("recursive", True)
        
        if not directory or not pattern:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32602,
                    "message": "Directory and pattern are required"
                }
            }
        
        try:
            import fnmatch
            
            full_dir = self.resolve_path(directory)
            
            if not full_dir.exists() or not full_dir.is_dir():
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": f"Directory not found: {full_dir}"
                    }
                }
            
            matches = []
            file_types = {}
            
            if recursive:
                for root, dirs, files in os.walk(full_dir):
                    for file in files:
                        if fnmatch.fnmatch(file, pattern):
                            file_path = str(Path(root) / file)
                            matches.append(file_path)
                            
                            # Count file types
                            ext = Path(file).suffix.lower()
                            file_types[ext] = file_types.get(ext, 0) + 1
            else:
                for item in full_dir.iterdir():
                    if item.is_file() and fnmatch.fnmatch(item.name, pattern):
                        matches.append(str(item))
                        ext = item.suffix.lower()
                        file_types[ext] = file_types.get(ext, 0) + 1
            
            # Create file type summary
            type_summary = []
            for ext, count in sorted(file_types.items()):
                if ext:
                    type_summary.append(f"{ext}: {count}")
                else:
                    type_summary.append(f"no extension: {count}")
            
            search_scope = "recursively" if recursive else "non-recursively"
            result_text = f"""🔍 Found {len(matches)} files matching '{pattern}' in {full_dir}
📊 Search: {search_scope}
📋 File types: {', '.join(type_summary) if type_summary else 'No files'}
⏰ Search time: {self.get_current_timestamp()} UTC

📄 Matches:
{chr(10).join(matches[:100])}"""
            
            if len(matches) > 100:
                result_text += f"\n... and {len(matches) - 100} more files"
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": result_text
                        }
                    ]
                }
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Error finding files: {str(e)}"
                }
            }

    async def run(self):
        """Run the MCP server using stdio transport"""
        logger.info(f"Starting Universal MCP File Server")
        logger.info(f"User: {self.current_user}")
        logger.info(f"Time: {self.current_time} UTC")
        logger.info(f"Supports: PDF, DOCX, XLSX, PPTX, TXT, code files, images, and more")
        
        while True:
            try:
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                request = json.loads(line)
                response = await self.handle_request(request)
                print(json.dumps(response), flush=True)
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    }
                }
                print(json.dumps(error_response), flush=True)
            except KeyboardInterrupt:
                logger.info("Server shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Universal MCP File Server - Supports ANY File Type")
    parser.add_argument(
        "--directory",
        "-d",
        default=None,
        help="Base directory for file operations (defaults to system root for full access)"
    )
    parser.add_argument(
        "--restrict",
        "-r",
        action="store_true",
        help="Restrict file operations to the specified directory only"
    )
    
    args = parser.parse_args()
    
    server = UniversalMCPFileServer(args.directory, args.restrict)
    asyncio.run(server.run())

if __name__ == "__main__":
    main()