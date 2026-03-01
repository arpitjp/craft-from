#!/usr/bin/env node
/** One-off script to capture browser console logs and screenshot from http://127.0.0.1:8000 */
import { webkit } from 'playwright';

const logs = [];

try {
  const browser = await webkit.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  page.on('console', (msg) => {
    const type = msg.type();
    const text = msg.text();
    const loc = msg.location();
    const entry = { type, text, url: loc?.url, line: loc?.lineNumber };
    logs.push(entry);
  });

  await page.goto('http://localhost:8766', { waitUntil: 'networkidle', timeout: 15000 });
  await page.waitForTimeout(5000);

  await page.screenshot({ path: '/Users/arpit.jain/p/craft-from/screenshot-8766.png', fullPage: false });
  await browser.close();

  console.log('=== ALL CONSOLE MESSAGES ===\n');
  logs.forEach((l) => {
    console.log(`[${l.type.toUpperCase()}] ${l.text}`);
    if (l.url) console.log(`  @ ${l.url}:${l.line || ''}`);
  });

  const errors = logs.filter((l) => l.type === 'error' || l.type === 'warning');
  if (errors.length) {
    console.log('\n=== ERRORS/WARNINGS ONLY ===');
    errors.forEach((l) => console.log(`[${l.type}] ${l.text}`));
  }
} catch (e) {
  console.error('Script error:', e.message);
  process.exit(1);
}
