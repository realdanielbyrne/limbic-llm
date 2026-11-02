"""limbic-llm: hippocampally inspired memory modules for sequence models.

Strategy I  (direct trisynaptic circuit): DentateGyrus, CA3Autoassociative,
            CA2Modulator, CA1Comparator, Subiculum, HippocampalBlock, HippocampalLM
Strategy II (Titans-hippocampal hybrid):  NeuralMemory, TitansHippocampalLayer,
            TitansHippocampalLM
Gated DeltaNet: gated_delta_scan, GatedDeltaNet, GatedDeltaMemory, GatedDeltaLM
Consolidation: PrioritizedReplayBuffer, Consolidator
"""

from .attention import TransformerLayer
from .ca1 import CA1Comparator
from .ca2 import CA2Modulator, TemporalContext
from .ca3 import CA3Autoassociative, EpisodicStore
from .dentate_gyrus import DentateGyrus
from .gated_delta import GatedDeltaMemory, GatedDeltaNet, GDNState, gated_delta_chunked, gated_delta_scan, kwta
from .hippocampal_block import HippocampalBlock, HippocampalOutput
from .models import BaselineLM, GatedDeltaLM, HippocampalLM, LMOutput, TitansHippocampalLM, lm_loss, novelty_alignment_loss
from .neural_memory import MemoryState, NeuralMemory
from .replay import Consolidator, PrioritizedReplayBuffer, orthogonal_projection, sequence_priority
from .subiculum import Subiculum
from .titans_hybrid import PersistentMemory, TitansHippocampalLayer, TitansOutput

__version__ = "0.1.0"
