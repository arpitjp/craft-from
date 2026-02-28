/**
 * Mineflayer bot: reads a .litematic schematic and builds it in a Minecraft server.
 *
 * Usage:
 *   node builder.js --host localhost --port 25565 --schematic ../output/output.litematic
 *
 * Requires a Minecraft Java server with online-mode=false (for bot auth)
 * or configure with Microsoft auth token.
 */

const mineflayer = require("mineflayer");
const { pathfinder } = require("mineflayer-pathfinder");
const { Schematic } = require("prismarine-schematic");
const fs = require("fs");
const path = require("path");

const args = parseArgs(process.argv.slice(2));

const HOST = args.host || "localhost";
const PORT = parseInt(args.port || "25565");
const USERNAME = args.username || "BuilderBot";
const SCHEMATIC_PATH = args.schematic || "../output/output.litematic";

async function main() {
  const bot = mineflayer.createBot({
    host: HOST,
    port: PORT,
    username: USERNAME,
    hideErrors: false,
  });

  bot.loadPlugin(pathfinder);

  bot.once("spawn", async () => {
    console.log(`Bot connected to ${HOST}:${PORT} as ${USERNAME}`);
    console.log(`Position: ${bot.entity.position}`);

    try {
      const schematicBuf = fs.readFileSync(path.resolve(SCHEMATIC_PATH));
      const schematic = await Schematic.read(schematicBuf, bot.version);
      console.log(`Loaded schematic: ${SCHEMATIC_PATH}`);

      const origin = bot.entity.position.floored();
      console.log(`Building at: ${origin}`);

      await bot.creative.flyTo(origin.offset(0, 5, 0));

      const blocks = [];
      const { start, end } = schematic;
      for (let y = start.y; y < end.y; y++) {
        for (let x = start.x; x < end.x; x++) {
          for (let z = start.z; z < end.z; z++) {
            const block = schematic.getBlock({ x, y, z });
            if (block && block.name !== "air") {
              blocks.push({
                pos: origin.offset(x - start.x, y - start.y, z - start.z),
                stateId: block.stateId,
              });
            }
          }
        }
      }

      console.log(`Placing ${blocks.length} blocks...`);
      let placed = 0;
      for (const { pos, stateId } of blocks) {
        try {
          await bot.creative.setBlock(pos, stateId);
          placed++;
          if (placed % 100 === 0) {
            console.log(`Progress: ${placed}/${blocks.length}`);
          }
        } catch (err) {
          // skip unplaceable blocks
        }
      }

      console.log(`Done. Placed ${placed}/${blocks.length} blocks.`);
    } catch (err) {
      console.error("Build failed:", err.message);
    }
  });

  bot.on("error", (err) => console.error("Bot error:", err));
  bot.on("kicked", (reason) => console.log("Kicked:", reason));
}

function parseArgs(argv) {
  const result = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith("--") && i + 1 < argv.length) {
      result[argv[i].slice(2)] = argv[i + 1];
      i++;
    }
  }
  return result;
}

main().catch(console.error);
