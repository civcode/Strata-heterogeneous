#pragma once
#include <cstddef>
#include <string>

namespace strata::core {
// Only an allocation failure with the original device restored permits prompt-reread fallback.
// Other runtime/device failures must abort the inference operation, even before snapshot writes.
enum class ConversationCopyPreparation { ready, unavailable, fatal };

// Acquire this thread's streams AND events on every requested device (-1 means its current device).
// Ownership remains in the existing thread-local handle set until thread exit, including resources acquired
// before a later device fails. Repeated calls reuse them; another thread cannot take them from the pool.
// Restores the original device. Does not submit copies, record events, wait on the legacy stream, or synchronize.
// This is resource preparation, not a promise that subsequent GPU operations cannot fail.
ConversationCopyPreparation conversation_copy_prepare(const int* devices, size_t count, std::string& error);
}
