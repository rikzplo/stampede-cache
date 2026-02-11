-- Atomic lock release: only delete if we own the lock
-- KEYS[1] = lock key
-- ARGV[1] = owner instance ID
-- Returns: 1 if deleted, 0 if not owned
local call, key = redis.call, KEYS[1]
if call("GET", key) == ARGV[1] then return call("DEL", key) end
return 0
