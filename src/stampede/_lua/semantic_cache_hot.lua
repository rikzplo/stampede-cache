-- Semantic cache hot-path: check Redis before PostgreSQL
-- KEYS[1] = exact hash cache key
-- KEYS[2] = stats key
-- ARGV[1] = TTL extension on hit (0 = no extension)
-- Returns: cached_value or nil

local keys, argv, call = KEYS, ARGV, redis.call
local k1, stats = keys[1], keys[2]
local cached = call("GET", k1)
if cached then
    call("HINCRBY", stats, "redis_hits", 1)
    local ttl = argv[1]
    if ttl ~= "0" and ttl ~= "" then call("EXPIRE", k1, ttl) end
    return cached
end
call("HINCRBY", stats, "redis_misses", 1)
return nil
