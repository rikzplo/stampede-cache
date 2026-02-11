-- Combined cache check + lock acquire in single round-trip
-- KEYS[1] = cache key
-- KEYS[2] = lock key
-- ARGV[1] = owner instance ID
-- ARGV[2] = lock TTL seconds
-- Returns: cached_value | "__LOCKED__" (we got lock) | "__WAIT__" (another has lock)
local call, keys, argv = redis.call, KEYS, ARGV
local cached = call("GET", keys[1])
if cached then return cached end
if call("SET", keys[2], argv[1], "NX", "EX", argv[2]) then return "__LOCKED__" end
return "__WAIT__"
