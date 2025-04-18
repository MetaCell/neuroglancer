export default {
  meta: {
    type: "problem",
    docs: {
      description:
        "Disallow `this` usage in class field initializers for dependent instance creation",
      recommended: false,
    },
    messages: {
      noThisReference:
        "Do not reference `this` in class field initializers when creating a new instance. Move it to the constructor instead.",
    },
    schema: [],
  },

  create(context) {
    function isInChildren(node, target) {
      if (node === target) {
        return true;
      }
      for (const key of Object.keys(node)) {
        if (key === "parent") {
          return false;
        }
        const value = node[key];

        if (Array.isArray(value)) {
          for (const el of value) {
            if (isInChildren(el)) return true;
          }
        } else if (value && typeof value === "object") {
          if (isInChildren(value)) return true;
        }
      }
      return false;
    }

    function parentCallExpression(node) {
      if (!node) {
        return undefined;
      }
      if (node.type === "CallExpression") {
        return node;
      }
      return parentCallExpression(node.parent);
    }

    function isInCalledArrowFunction(node) {
      if (!node) {
        return false;
      }
      if (
        node.type === "ArrowFunctionExpression" ||
        node.type === "FunctionExpression"
      ) {
        const parentCallExpr = parentCallExpression(node);
        if (!parentCallExpr) {
          return false;
        }
        for (const arg of parentCallExpr.arguments) {
          if (isInChildren(arg, node)) {
            return true;
          }
        }
        return parentCallExpr && isInChildren(parentCallExpr.callee, node);
      }
      return isInCalledArrowFunction(node.parent);
    }

    function isInArrowFunction(node) {
      if (!node) {
        return false;
      }
      if (node.type === "ArrowFunctionExpression") {
        return true;
      }
      return isInArrowFunction(node.parent);
    }

    function containsThisExpression(node) {
      if (!node || typeof node !== "object") return false;
      if (
        node.type === "ArrowFunctionExpression" ||
        node.type === "FunctionExpression"
      ) {
        return false;
      }

      if (node.type === "ThisExpression") {
        // We report an error if "this" is in a lambda directly called:
        //   (() => this.foo)()
        // of if it's not in an arrow function
        //   this.foo
        //   () => this.foo  -- this will not trigger an error as the execution of foo is delayed for later
        // return true
        return isInCalledArrowFunction(node) || !isInArrowFunction(node);
      }

      if (node.type === "CallExpression") {
        // If we have a call, we check the presence of "this" in the arguments
        for (const argument of node.arguments) {
          if (containsThisExpression(argument)) return true;
        }
        // We check if the callee is not a lambda that is directly called
        const callee = node.callee;
        if (
          callee.type === "ArrowFunctionExpression" &&
          containsThisExpression(callee.body)
        ) {
          return true;
        }
        if (containsThisExpression(callee)) {
          return true;
        }
        // We visited the arguments and the callee, no need to go further
        return false;
      }

      // We walk down the AST for other nodes
      for (const key of Object.keys(node)) {
        if (key === "parent") {
          return false;
        }
        const value = node[key];

        if (Array.isArray(value)) {
          for (const el of value) {
            if (containsThisExpression(el)) return true;
          }
        } else if (value && typeof value === "object") {
          if (containsThisExpression(value)) return true;
        }
      }

      return false;
    }

    return {
      PropertyDefinition(node) {
        if (!node.value) return;
        if (containsThisExpression(node.value)) {
          context.report({
            node,
            messageId: "noThisReference",
          });
        }
      },
    };
  },
};
