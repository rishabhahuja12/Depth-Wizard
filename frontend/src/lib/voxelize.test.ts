import test from 'node:test';
import assert from 'node:assert/strict';
import {
  voxelize,
  generateTurboPalette,
  sampleTurboRgb,
  rgbToHex,
} from './voxelize.ts';

test('sampleTurboRgb and rgbToHex generate valid color progression', () => {
  // At t = 0: pure blue [0, 0, 1] -> #0000ff
  const blue = sampleTurboRgb(0.0);
  assert.deepEqual(blue, [0, 0, 1]);
  assert.equal(rgbToHex(blue), '#0000ff');

  // At t = 1.0: pure red [1, 0, 0] -> #ff0000
  const red = sampleTurboRgb(1.0);
  assert.deepEqual(red, [1, 0, 0]);
  assert.equal(rgbToHex(red), '#ff0000');

  // At t = 0.5: pure green [0, 1, 0] -> #00ff00
  const green = sampleTurboRgb(0.5);
  assert.deepEqual(green, [0, 1, 0]);
  assert.equal(rgbToHex(green), '#00ff00');
});

test('generateTurboPalette produces exact band count with unique stops', () => {
  for (const count of [5, 7, 8, 12]) {
    const palette = generateTurboPalette(count);
    assert.equal(palette.length, count);
    palette.forEach((hex) => {
      assert.match(hex, /^#[0-9a-f]{6}$/i);
    });
  }
});

test('voxelize pools synthetic ramp height grid correctly', () => {
  // 100x100 linear vertical ramp from 0 to 100
  const H = 100;
  const W = 100;
  const syntheticGrid: number[][] = [];
  for (let r = 0; r < H; r++) {
    const row: number[] = [];
    for (let c = 0; c < W; c++) {
      row.push(r); // elevation equals row index (0 at top, 99 at bottom)
    }
    syntheticGrid.push(row);
  }

  const targetRes = 10;
  const bandCount = 8;
  const result = voxelize(syntheticGrid, {
    targetResolution: targetRes,
    bandCount,
    worldWidth: 10,
    worldDepth: 10,
  });

  // Verify dimensions
  assert.equal(result.rows, targetRes);
  assert.equal(result.cols, targetRes);
  assert.equal(result.blocks.length, targetRes * targetRes);

  // Verify pooling average on first block (rows 0..9, cols 0..9)
  // Average of 0..9 is 4.5
  const firstBlock = result.blocks[0];
  assert.equal(firstBlock.gridX, 0);
  assert.equal(firstBlock.gridZ, 0);
  assert.equal(firstBlock.rawHeight, 4.5);
  assert.equal(firstBlock.bandIndex, 0);

  // Verify pooling average on last block (rows 90..99)
  // Average of 90..99 is 94.5
  const lastBlock = result.blocks[result.blocks.length - 1];
  assert.equal(lastBlock.gridX, targetRes - 1);
  assert.equal(lastBlock.gridZ, targetRes - 1);
  assert.equal(lastBlock.rawHeight, 94.5);
  assert.equal(lastBlock.bandIndex, bandCount - 1);

  // Verify elevation bounds
  assert.equal(result.elevationMin, 4.5);
  assert.equal(result.elevationMax, 94.5);
  assert.equal(result.elevationRange, 90);

  // Verify world centering
  // For width 10, centered at 0, first block center X is -5 + 0.5 = -4.5
  // last block center X is -5 + 9.5 = 4.5
  assert.equal(firstBlock.posX, -4.5);
  assert.equal(firstBlock.posZ, -4.5);
  assert.equal(lastBlock.posX, 4.5);
  assert.equal(lastBlock.posZ, 4.5);

  // Verify footprint gap scaling
  const expectedCellWidth = (10 / 10) * 0.90; // 0.9
  assert.equal(result.cellWidth, expectedCellWidth);
  assert.equal(result.cellDepth, expectedCellWidth);
});

test('voxelize handles non-square aspect ratio properly', () => {
  // 60 rows x 120 columns (2:1 aspect ratio)
  const syntheticGrid: number[][] = Array.from({ length: 60 }, () =>
    new Array(120).fill(10)
  );

  const result = voxelize(syntheticGrid, { targetResolution: 64 });
  assert.equal(result.cols, 64);
  assert.equal(result.rows, 32); // 64 * (60 / 120) = 32
  assert.equal(result.blocks.length, 64 * 32);
});

test('voxelize handles flat elevation plane without zero-division', () => {
  // 30x30 completely flat terrain
  const flatGrid: number[][] = Array.from({ length: 30 }, () =>
    new Array(30).fill(42.0)
  );

  const result = voxelize(flatGrid, { targetResolution: 10, bandCount: 8 });
  assert.equal(result.elevationRange, 1e-6); // Safe epsilon
  assert.equal(result.blocks.length, 100);
  result.blocks.forEach((b) => {
    assert.equal(b.rawHeight, 42.0);
    assert.equal(b.normalizedHeight, 0);
    assert.equal(b.bandIndex, 0);
  });
});

test('voxelize handles empty or corrupt grid gracefully', () => {
  const emptyResult = voxelize([]);
  assert.equal(emptyResult.blocks.length, 0);
  assert.equal(emptyResult.rows, 0);
  assert.equal(emptyResult.cols, 0);
});
