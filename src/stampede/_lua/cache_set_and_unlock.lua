-- Atomic cache set + lock release (only if we own the lock)
-- KEYS[1] = cache key
-- KEYS[2] = lock key
-- ARGV[1] = value to cache
-- ARGV[2] = cache TTL seconds
-- ARGV[3] = our owner ID
-- Returns: 1 if set, 0 if lock not owned
local call, keys, argv = redis.call, KEYS, ARGV
local lock = keys[2]
if call("GET", lock) ~= argv[3] then return 0 end
call("SETEX", keys[1], argv[2], argv[1])
call("DEL", lock)
return 1
