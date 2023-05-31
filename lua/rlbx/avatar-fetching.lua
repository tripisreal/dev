local players = game:GetService("Players"):GetPlayers()

for i, player in ipairs(players) do
    local userId = player.UserId
    local hd = player:GetHumanoidDescription()

    print(userId, hd:ToString())
end


-- local hd = Players:GetHumanoidDescriptionFromUserId(userId)