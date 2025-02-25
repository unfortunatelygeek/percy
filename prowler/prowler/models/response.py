from typing import List
from enum import Enum
from pydantic import BaseModel, Field

class VulnType(str, Enum):
    LFI = "LFI"
    RCE = "RCE"
    SSRF = "SSRF"
    AFO = "AFO"
    SQLI = "SQLI"
    XSS = "XSS"
    IDOR = "IDOR"

class ContextCode(BaseModel):
    name: str = Field(description="Function or class name")
    reason: str = Field(description="Brief reason why this function's code is needed for analysis")
    code_line: str = Field(description="The single line of code where this context object is referenced.")

class Response(BaseModel):
    scratchpad: str = Field(description="Step-by-step analysis process, in plaintext without line breaks.")
    analysis: str = Field(description="Final vulnerability analysis in plaintext without line breaks.")
    poc: str = Field(description="Proof-of-concept exploit, if applicable.")
    confidence_score: int = Field(
        description="0-10, where 0 is no confidence and 10 is absolute certainty due to full understanding of user input to server output code path."
    )
    vulnerability_types: List[VulnType] = Field(description="List of identified vulnerability types.")
    context_code: List[ContextCode] = Field(
        description="List of relevant functions or classes requested for analysis."
    )
