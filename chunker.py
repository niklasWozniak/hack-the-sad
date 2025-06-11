import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import mimetypes

@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    file_path: str
    chunk_index: int
    
    def __post_init__(self):
        # Generate unique chunk ID if not provided
        if not self.chunk_id:
            content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
            self.chunk_id = f"{Path(self.file_path).stem}_{self.chunk_index}_{content_hash}"

class DocumentChunker:
    """Handles chunking of various document types"""
    
    def __init__(self, 
                 text_chunk_size: int = 1000,
                 text_overlap: int = 200,
                 code_chunk_size: int = 1500,
                 code_overlap: int = 100):
        self.text_chunk_size = text_chunk_size
        self.text_overlap = text_overlap
        self.code_chunk_size = code_chunk_size
        self.code_overlap = code_overlap
        
        # File type mappings
        self.code_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala',
            '.r', '.sql', '.sh', '.bash', '.ps1', '.yaml', '.yml', '.json',
            '.xml', '.html', '.css', '.scss', '.less', '.vue', '.svelte'
        }
        
        self.text_extensions = {
            '.txt', '.md', '.rst', '.tex', '.pdf', '.doc', '.docx'
        }
    
    def get_file_type(self, file_path: str) -> str:
        """Determine if file is code, text, or other"""
        ext = Path(file_path).suffix.lower()
        if ext in self.code_extensions:
            return 'code'
        elif ext in self.text_extensions:
            return 'text'
        else:
            # Try to guess from mime type
            mime_type, _ = mimetypes.guess_type(file_path)
            if mime_type and mime_type.startswith('text/'):
                return 'text'
            return 'unknown'
    
    def chunk_file(self, file_path: str) -> List[DocumentChunk]:
        """Main method to chunk a file based on its type"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []
        
        if not content.strip():
            return []
        
        file_type = self.get_file_type(file_path)
        base_metadata = self._get_base_metadata(file_path, file_type)
        
        if file_type == 'code':
            return self._chunk_code(content, file_path, base_metadata)
        elif file_type == 'text':
            return self._chunk_text(content, file_path, base_metadata)
        else:
            # Default to text chunking for unknown types
            return self._chunk_text(content, file_path, base_metadata)
    
    def _get_base_metadata(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Generate base metadata for a file"""
        path_obj = Path(file_path)
        stat = path_obj.stat()
        
        return {
            'file_path': str(path_obj),
            'file_name': path_obj.name,
            'file_extension': path_obj.suffix,
            'file_type': file_type,
            'file_size': stat.st_size,
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'directory': str(path_obj.parent),
        }
    
    def _chunk_code(self, content: str, file_path: str, base_metadata: Dict) -> List[DocumentChunk]:
        """Chunk code files by functions, classes, and logical blocks"""
        chunks = []
        lines = content.split('\n')
        
        # Detect programming language for better parsing
        language = self._detect_language(file_path)
        base_metadata['language'] = language
        
        if language in ['python', 'javascript', 'typescript', 'java', 'cpp']:
            chunks = self._chunk_structured_code(content, file_path, base_metadata, language)
        else:
            # Fall back to size-based chunking for other languages
            chunks = self._chunk_by_size(content, file_path, base_metadata, 
                                       self.code_chunk_size, self.code_overlap)
        
        return chunks
    
    def _chunk_structured_code(self, content: str, file_path: str, 
                             base_metadata: Dict, language: str) -> List[DocumentChunk]:
        """Chunk code by functions and classes"""
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_function = None
        current_class = None
        chunk_index = 0
        
        # Language-specific patterns
        patterns = {
            'python': {
                'function': r'^\s*def\s+(\w+)',
                'class': r'^\s*class\s+(\w+)',
                'import': r'^\s*(import|from)\s+'
            },
            'javascript': {
                'function': r'^\s*(function\s+\w+|const\s+\w+\s*=.*=>|\w+\s*:\s*function)',
                'class': r'^\s*class\s+(\w+)',
                'import': r'^\s*(import|export)\s+'
            },
            'java': {
                'function': r'^\s*(public|private|protected).*\w+\s*\(',
                'class': r'^\s*(public|private)?\s*class\s+(\w+)',
                'import': r'^\s*import\s+'
            }
        }
        
        function_pattern = patterns.get(language, {}).get('function', r'^\s*\w+.*\(')
        class_pattern = patterns.get(language, {}).get('class', r'^\s*class\s+\w+')
        
        for i, line in enumerate(lines):
            current_chunk.append(line)
            
            # Check for new function/class
            if re.match(function_pattern, line):
                current_function = line.strip()
            elif re.match(class_pattern, line):
                current_class = line.strip()
            
            # Create chunk when we have enough content or hit a new major block
            if (len('\n'.join(current_chunk)) > self.code_chunk_size or 
                (i > 0 and re.match(function_pattern, line) and len(current_chunk) > 10)):
                
                if len(current_chunk) > 1:  # Don't create empty chunks
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata.update({
                        'current_function': current_function,
                        'current_class': current_class,
                        'start_line': i - len(current_chunk) + 1,
                        'end_line': i,
                        'chunk_type': 'code_block'
                    })
                    
                    chunk = DocumentChunk(
                        content='\n'.join(current_chunk),
                        metadata=chunk_metadata,
                        chunk_id='',
                        file_path=file_path,
                        chunk_index=chunk_index
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Keep some overlap
                overlap_lines = max(1, self.code_overlap // 20)  # Rough lines for overlap
                current_chunk = current_chunk[-overlap_lines:] if overlap_lines < len(current_chunk) else []
        
        # Add remaining content
        if current_chunk:
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'current_function': current_function,
                'current_class': current_class,
                'start_line': len(lines) - len(current_chunk),
                'end_line': len(lines),
                'chunk_type': 'code_block'
            })
            
            chunk = DocumentChunk(
                content='\n'.join(current_chunk),
                metadata=chunk_metadata,
                chunk_id='',
                file_path=file_path,
                chunk_index=chunk_index
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_text(self, content: str, file_path: str, base_metadata: Dict) -> List[DocumentChunk]:
        """Chunk text documents by paragraphs and semantic breaks"""
        chunks = []
        
        # First try to chunk by semantic boundaries (paragraphs, sections)
        if self._is_markdown(file_path):
            chunks = self._chunk_markdown(content, file_path, base_metadata)
        else:
            # Fall back to paragraph-based chunking
            chunks = self._chunk_by_paragraphs(content, file_path, base_metadata)
        
        return chunks
    
    def _chunk_markdown(self, content: str, file_path: str, base_metadata: Dict) -> List[DocumentChunk]:
        """Chunk markdown by headers and sections"""
        chunks = []
        sections = re.split(r'\n(?=#{1,6}\s)', content)
        
        chunk_index = 0
        for section in sections:
            if not section.strip():
                continue
                
            # Extract header if present
            header_match = re.match(r'^(#{1,6})\s+(.+)', section)
            header_level = len(header_match.group(1)) if header_match else 0
            header_text = header_match.group(2) if header_match else None
            
            # Split large sections further if needed
            if len(section) > self.text_chunk_size:
                sub_chunks = self._chunk_by_size(section, file_path, base_metadata,
                                               self.text_chunk_size, self.text_overlap)
                for sub_chunk in sub_chunks:
                    sub_chunk.metadata.update({
                        'header_text': header_text,
                        'header_level': header_level,
                        'chunk_type': 'markdown_section'
                    })
                    sub_chunk.chunk_index = chunk_index
                    chunks.append(sub_chunk)
                    chunk_index += 1
            else:
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'header_text': header_text,
                    'header_level': header_level,
                    'chunk_type': 'markdown_section'
                })
                
                chunk = DocumentChunk(
                    content=section,
                    metadata=chunk_metadata,
                    chunk_id='',
                    file_path=file_path,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    def _chunk_by_paragraphs(self, content: str, file_path: str, base_metadata: Dict) -> List[DocumentChunk]:
        """Chunk text by paragraphs, combining small ones"""
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for paragraph in paragraphs:
            para_size = len(paragraph)
            
            if current_size + para_size > self.text_chunk_size and current_chunk:
                # Create chunk from current paragraphs
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_type': 'text_paragraphs',
                    'paragraph_count': len(current_chunk)
                })
                
                chunk = DocumentChunk(
                    content='\n\n'.join(current_chunk),
                    metadata=chunk_metadata,
                    chunk_id='',
                    file_path=file_path,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
                
                # Start new chunk with overlap
                if self.text_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-1]
                    if len(overlap_text) <= self.text_overlap:
                        current_chunk = [overlap_text]
                        current_size = len(overlap_text)
                    else:
                        current_chunk = []
                        current_size = 0
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(paragraph)
            current_size += para_size
        
        # Add remaining content
        if current_chunk:
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                'chunk_type': 'text_paragraphs',
                'paragraph_count': len(current_chunk)
            })
            
            chunk = DocumentChunk(
                content='\n\n'.join(current_chunk),
                metadata=chunk_metadata,
                chunk_id='',
                file_path=file_path,
                chunk_index=chunk_index
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_size(self, content: str, file_path: str, base_metadata: Dict,
                      chunk_size: int, overlap: int) -> List[DocumentChunk]:
        """Fallback chunking by character size with overlap"""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # Try to break at word boundary
            if end < len(content):
                # Look for last space/newline in the chunk
                last_break = max(
                    content.rfind(' ', start, end),
                    content.rfind('\n', start, end)
                )
                if last_break > start:
                    end = last_break
            
            chunk_content = content[start:end].strip()
            if chunk_content:
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_type': 'size_based',
                    'char_start': start,
                    'char_end': end
                })
                
                chunk = DocumentChunk(
                    content=chunk_content,
                    metadata=chunk_metadata,
                    chunk_id='',
                    file_path=file_path,
                    chunk_index=chunk_index
                )
                chunks.append(chunk)
                chunk_index += 1
            
            start = max(start + 1, end - overlap)
        
        return chunks
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        ext = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.jsx': 'javascript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust',
            '.swift': 'swift',
            '.kt': 'kotlin'
        }
        return language_map.get(ext, 'unknown')
    
    def _is_markdown(self, file_path: str) -> bool:
        """Check if file is markdown"""
        return Path(file_path).suffix.lower() in ['.md', '.markdown']

# Example usage and testing
def main():
    chunker = DocumentChunker()
    
    # Example: chunk a Python file
    test_file = "test.txt"
    if os.path.exists(test_file):
        chunks = chunker.chunk_file(test_file)
        print(f"Created {len(chunks)} chunks from {test_file}")
        for i, chunk in enumerate(chunks):  # Show first 3 chunks
            print(f"\n--- Chunk {i+1} ---")
            print(f"ID: {chunk.chunk_id}")
            print(f"Type: {chunk.metadata.get('chunk_type')}")
            print(f"Content preview: {chunk.content[:200]}...")

if __name__ == "__main__":
    main()