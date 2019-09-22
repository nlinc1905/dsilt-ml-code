SELECT datetime(c.created_utc, 'unixepoch') AS created_utc, c.subreddit, c.author, c.score, c.body
FROM May2015 c
WHERE c.subreddit = 'NFL_Draft'
	AND datetime(c.created_utc, 'unixepoch') BETWEEN &Date1 AND &Date2