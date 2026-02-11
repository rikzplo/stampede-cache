-- Atomic cache get-or-set with thundering herd prevention
-- KEYS[1] = cache key
-- KEYS[2] = lock key
-- ARGV[1] = owner instance ID
-- ARGV[2] = lock TTL seconds
-- ARGV[3] = processing marker
-- Returns: cached_value | "__COMPUTE__" (we got lock) | "__WAIT__" (another has lock)

local call, argv = redis.call, ARGV
local cached = call("GET", KEYS[1])
if cached then return cached end
if call("SET", KEYS[2], argv[1], "NX", "EX", argv[2]) then return "__COMPUTE__" end
return "__WAIT__"
